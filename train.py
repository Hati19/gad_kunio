from __future__ import print_function
import tensorflow as tf
# noinspection PyPep8Naming
import TensorFlow_utils as tf_utils
import numpy as np
#import sys
import os
#from sklearn.utils import shuffle
#from skimage.transform import resize
from skimage.io import imsave
#import time
import collections
import matplotlib.pyplot as plt
plt.switch_backend('agg') # Very Important in R Markdown
import matplotlib.gridspec as gridspec
from PIL import Image
from skimage.io import imsave, imread
#import pickle
from skimage import filters
import timeit




image_ch=5
gen_c=32
init_lr= 2e-4
beta1= 0.5
#iters=100
batch_size=8
img_rows=256
img_cols=256
num_epoch=1000
input_training_dir='/users/PCCF0019/ccf0071/Downloads/gald_seg/data/'

model_out_dir='weights/'


def IOU_(y_pred, y_true):
    """Returns a (approx) IOU score
    intesection = y_pred.flatten() * y_true.flatten()
    Then, IOU = 2 * intersection / (y_pred.sum() + y_true.sum() + 1e-7) + 1e-7
    Args:
        y_pred (4-D array): (N, H, W, 1)
        y_true (4-D array): (N, H, W, 1)
    Returns:
        float: IOU score
    """
    H, W, _ = y_pred.get_shape().as_list()[1:]

    pred_flat = tf.reshape(y_pred, [-1, H * W])
    true_flat = tf.reshape(y_true, [-1, H * W])

    intersection = 2 * tf.reduce_sum(pred_flat * true_flat, axis=1) + 1e-7
    denominator = tf.reduce_sum(
        pred_flat, axis=1) + tf.reduce_sum(
            true_flat, axis=1) + 1e-7

    return tf.reduce_mean(intersection / denominator)

def dice_coef(y_true, y_pred): 
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice(im1, im2, empty_score=1.0):  #different function copied from net
    """
    Computes the Dice coefficient, a measure of set similarity.
    Parameters
    ----------
    im1 : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    im2 : array-like, bool
        Any other array of identical size. If not boolean, will be converted.
    Returns
    -------
    dice : float
        Dice coefficient as a float on range [0,1].
        Maximum similarity = 1
        No similarity = 0
        Both are empty (sum eq to zero) = empty_score
        
    Notes
    -----
    The order of inputs for `dice` is irrelevant. The result will be
    identical if `im1` and `im2` are switched.
    """
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)
    #print (im1.sum() )
    #print (im2.sum() )
    return 2. * intersection.sum() / im_sum


def dice_coef_loss(y_true, y_pred):
    return -IOU_(y_pred,y_true)
    
def threshold_by_otsu(pred_vessels,  flatten=False):
    # cut by otsu threshold
    threshold = filters.threshold_otsu(pred_vessels)
    pred_vessels_bin = np.zeros(pred_vessels.shape)
    pred_vessels_bin[pred_vessels >= threshold] = 1

    if flatten:
        return pred_vessels_bin.flatten()
    else:
        return pred_vessels_bin

def generator(data, name='g_'):
    with tf.variable_scope(name):
        # conv1: (N, 640, 640, 1) -> (N, 320, 320, 32)
        conv1 = tf_utils.conv2d(data, gen_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv1_conv1')
        conv1 = tf_utils.batch_norm(conv1, name='conv1_batch1', _ops=_gen_train_ops)
        conv1 = tf.nn.relu(conv1, name='conv1_relu1')
        conv1 = tf_utils.conv2d(conv1, gen_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv1_conv2')
        conv1 = tf_utils.batch_norm(conv1, name='conv1_batch2', _ops=_gen_train_ops)
        conv1 = tf.nn.relu(conv1, name='conv1_relu2')
        pool1 = tf_utils.max_pool_2x2(conv1, name='maxpool1')

        # conv2: (N, 320, 320, 32) -> (N, 160, 160, 64)
        conv2 = tf_utils.conv2d(pool1, 2*gen_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv2_conv1')
        conv2 = tf_utils.batch_norm(conv2, name='conv2_batch1', _ops=_gen_train_ops)
        conv2 = tf.nn.relu(conv2, name='conv2_relu1')
        conv2 = tf_utils.conv2d(conv2, 2*gen_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv2_conv2')
        conv2 = tf_utils.batch_norm(conv2, name='conv2-batch2', _ops=_gen_train_ops)
        conv2 = tf.nn.relu(conv2, name='conv2_relu2')
        pool2 = tf_utils.max_pool_2x2(conv2, name='maxpool2')

        # conv3: (N, 160, 160, 64) -> (N, 80, 80, 128)
        conv3 = tf_utils.conv2d(pool2, 4*gen_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv3_conv1')
        conv3 = tf_utils.batch_norm(conv3, name='conv3_batch1', _ops=_gen_train_ops)
        conv3 = tf.nn.relu(conv3, name='conv3_relu1')
        conv3 = tf_utils.conv2d(conv3, 4*gen_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv3_conv2')
        conv3 = tf_utils.batch_norm(conv3, name='conv3_batch2', _ops=_gen_train_ops)
        conv3 = tf.nn.relu(conv3, name='conv3_relu2')
        pool3 = tf_utils.max_pool_2x2(conv3, name='maxpool3')

        # conv4: (N, 80, 80, 128) -> (N, 40, 40, 256)
        conv4 = tf_utils.conv2d(pool3, 8*gen_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv4_conv1')
        conv4 = tf_utils.batch_norm(conv4, name='conv4_batch1', _ops=_gen_train_ops)
        conv4 = tf.nn.relu(conv4, name='conv4_relu1')
        conv4 = tf_utils.conv2d(conv4, 8*gen_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv4_conv2')
        conv4 = tf_utils.batch_norm(conv4, name='conv4_batch2', _ops=_gen_train_ops)
        conv4 = tf.nn.relu(conv4, name='conv4_relu2')
        pool4 = tf_utils.max_pool_2x2(conv4, name='maxpool4')

        # conv5: (N, 40, 40, 256) -> (N, 40, 40, 512)
        conv5 = tf_utils.conv2d(pool4, 16*gen_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv5_conv1')
        conv5 = tf_utils.batch_norm(conv5, name='conv5_batch1', _ops=_gen_train_ops)
        conv5 = tf.nn.relu(conv5, name='conv5_relu1')
        conv5 = tf_utils.conv2d(conv5, 16*gen_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv5_conv2')
        conv5 = tf_utils.batch_norm(conv5, name='conv5_batch2', _ops=_gen_train_ops)
        conv5 = tf.nn.relu(conv5, name='conv5_relu2')

        # conv6: (N, 40, 40, 512) -> (N, 80, 80, 256)
        up1 = tf_utils.upsampling2d(conv5, size=(2, 2), name='conv6_up')
        conv6 = tf.concat([up1, conv4], axis=3, name='conv6_concat')
        conv6 = tf_utils.conv2d(conv6, 8*gen_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv6_conv1')
        conv6 = tf_utils.batch_norm(conv6, name='conv6_batch1', _ops=_gen_train_ops)
        conv6 = tf.nn.relu(conv6, name='conv6_relu1')
        conv6 = tf_utils.conv2d(conv6, 8*gen_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv6_conv2')
        conv6 = tf_utils.batch_norm(conv6, name='conv6_batch2', _ops=_gen_train_ops)
        conv6 = tf.nn.relu(conv6, name='conv6_relu2')

        # conv7: (N, 80, 80, 256) -> (N, 160, 160, 128)
        up2 = tf_utils.upsampling2d(conv6, size=(2, 2), name='conv7_up')
        conv7 = tf.concat([up2, conv3], axis=3, name='conv7_concat')
        conv7 = tf_utils.conv2d(conv7, 4*gen_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv7_conv1')
        conv7 = tf_utils.batch_norm(conv7, name='conv7_batch1', _ops=_gen_train_ops)
        conv7 = tf.nn.relu(conv7, name='conv7_relu1')
        conv7 = tf_utils.conv2d(conv7, 4*gen_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv7_conv2')
        conv7 = tf_utils.batch_norm(conv7, name='conv7_batch2', _ops=_gen_train_ops)
        conv7 = tf.nn.relu(conv7, name='conv7_relu2')

        # conv8: (N, 160, 160, 128) -> (N, 320, 320, 64)
        up3 = tf_utils.upsampling2d(conv7, size=(2, 2), name='conv8_up')
        conv8 = tf.concat([up3, conv2], axis=3, name='conv8_concat')
        conv8 = tf_utils.conv2d(conv8, 2*gen_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv8_conv1')
        conv8 = tf_utils.batch_norm(conv8, name='conv8_batch1', _ops=_gen_train_ops)
        conv8 = tf.nn.relu(conv8, name='conv8_relu1')
        conv8 = tf_utils.conv2d(conv8, 2*gen_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv8_conv2')
        conv8 = tf_utils.batch_norm(conv8, name='conv8_batch2', _ops=_gen_train_ops)
        conv8 = tf.nn.relu(conv8, name='conv8_relu2')

        # conv9: (N, 320, 320, 64) -> (N, 640, 640, 32)
        up4 = tf_utils.upsampling2d(conv8, size=(2, 2), name='conv9_up')
        conv9 = tf.concat([up4, conv1], axis=3, name='conv9_concat')
        conv9 = tf_utils.conv2d(conv9, gen_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv9_conv1')
        conv9 = tf_utils.batch_norm(conv9, name='conv9_batch1', _ops=_gen_train_ops)
        conv9 = tf.nn.relu(conv9, name='conv9_relu1')
        conv9 = tf_utils.conv2d(conv9, gen_c, k_h=3, k_w=3, d_h=1, d_w=1, name='conv9_conv2')
        conv9 = tf_utils.batch_norm(conv9, name='conv9_batch2', _ops=_gen_train_ops)
        conv9 = tf.nn.relu(conv9, name='conv9_relu2')

        # output layer: (N, 640, 640, 32) -> (N, 640, 640, 1)
        output = tf_utils.conv2d(conv9, 1, k_h=1, k_w=1, d_h=1, d_w=1, name='conv_output')

        return tf.nn.sigmoid(output)
        
def get_data(input_file_name_list):
  
    images=input_file_name_list
    total = len(images) 
    imgs = np.ndarray((total, img_rows, img_cols,image_ch), dtype=np.float32)
    imgs_mask = np.ndarray((total,img_rows, img_cols), dtype=np.float32)
    i=0
    for image_name in images:
        
        if 'mask'  in image_name:
            continue
        
        image_mask_name = image_name.split('.')[0] + '_mask.npy'
        img = np.load(image_name)
        #img=np.squeeze(img, axis=0)
        #mean=np.mean(img)
        #print(mean)
       
        #adjusted_stddev = max(np.std(img), 1.0/np.sqrt(img_rows* img_cols*image_ch))
        #print(adjusted_stddev)
        #img=(img - mean) / adjusted_stddev
        img_mask = np.load(image_mask_name)
        #img_mask=np.squeeze(img_mask, axis=0)
        img = np.array([img])
        img_mask = np.array([img_mask])
        imgs[i] = img
        imgs_mask[i] = img_mask
        i+=1
        #if(i%160)
        
    
    #print(imgs.shape)
    #print(imgs_mask.shape)
    
    return imgs, imgs_mask[..., np.newaxis]
    
def get_train_file_names(input_file_name):
    images = sorted(os.listdir(input_file_name))
    train_id=[]
    train_mask_id=[]
    for image_name in images:
        if 'mask'  in image_name:
            #print(os.path.join(input_file_name, image_name))
            continue
        image_mask_name = image_name.split('.')[0] + '_mask.npy'
        train_id.append(os.path.join(input_file_name, image_name))
        #print(os.path.join(input_file_name, image_name))
        train_mask_id.append(os.path.join(input_file_name, image_mask_name))            
          
    
    return train_id,train_mask_id   
    
    

  
       
# Construct model
print('Model Initialization start')
X = tf.placeholder(tf.float32, shape=[None,  img_rows, img_cols, image_ch], name='image')
Y = tf.placeholder(tf.float32, shape=[None,  img_rows, img_cols,1], name='vessel')
_gen_train_ops=[]
y_pred = generator(X)
print('defining Cost start')
# Define loss and optimizer
#cost = tf.reduce_mean(tf.square(g_samples-Y))
cost=dice_coef_loss(Y,y_pred)
gen_op = tf.train.AdamOptimizer(learning_rate=init_lr, beta1=beta1).minimize(cost)

gen_ops = [gen_op] + _gen_train_ops
gen_optim = tf.group(*gen_ops)

print('Model Initialization done')


# start training data load
total_id,total_mask_id=get_train_file_names(input_training_dir)
start_time = timeit.default_timer()
#total_id=total_id[1:1000]
print(len(total_id))
np.random.shuffle(total_id)
train_id_temp=total_id[:int(len(total_id)*.9)]
test_id=total_id[int(len(total_id)*.9):]
with open('test_id.txt', 'w') as filehandle:
    for listitem in test_id:
        filehandle.write('%s\n' % listitem)
np.random.shuffle(train_id_temp)
#label=int((len(train_id_temp)/320)*.8)*320
label=int((len(train_id_temp))*.7)
print(label)
train_id=train_id_temp[:label]
val_id=train_id_temp[label:]

     
print(len(train_id))
print(len(val_id))
print(len(test_id))


best_auc_sum=0
# --- start session ---
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for iter in range(num_epoch):
        
        # train
        train_loss=[]
        for current_batch_index in range(0,len(train_id),batch_size):
            current_batch_id  = [train_id[i] for i in range(current_batch_index,min(len(train_id),current_batch_index+batch_size))]
            #print(len(current_batch_id))
            current_batch,current_label=get_data(current_batch_id)
            #current_batch=current_batch[..., np.newaxis]
            #current_label=current_label[..., np.newaxis]
            sess_results = sess.run([cost,gen_optim],feed_dict={X:current_batch,Y:current_label})
            train_loss.append(sess_results[0])
            #print(len(sess_results))
        print(' Iter: ', iter, " Cost:  %.3f"% np.mean(train_loss))
        #print('\n-----------------------')
        np.random.shuffle(train_id)
        #print('\n------Validation-----------------')
        dice_loss=[]
        for current_batch_index in range(0,len(val_id),batch_size):
            current_batch_id  = [val_id[i] for i in range(current_batch_index,min(len(val_id),current_batch_index+batch_size))]
            #print(len(current_batch_id))
            current_batch,current_label=get_data(current_batch_id)
            #current_batch=current_batch[..., np.newaxis]
            #current_label=current_label[..., np.newaxis]
            sess_results = sess.run([cost],feed_dict={X:current_batch,Y:current_label})
            dice_loss.append(sess_results[0])
            #print(' Iter: ', iter, " Cost:  %.3f"% sess_results[0],end='\r')
        #print('\n-------Validation loss----------------')
        print(' Iter: ', iter, " Validation loss:  %.3f"% np.mean(dice_loss))
        #print(np.mean(dice_loss))
        #print('\n-----------------------')
        auc_sum=-np.mean(dice_loss)
        flag=0
        if auc_sum > .7:
            
        #if 1:
            #print('\n-----------------------')
            #print('\n--------save model-----')
            #print('\n-----------------------')
            if best_auc_sum < auc_sum:
                flag=1
                best_auc_sum = auc_sum
                saver = tf.train.Saver()
                model_name = "iter_{}_auc_sum_{:.3}".format(iter, best_auc_sum)
                saver.save(sess, os.path.join(model_out_dir, model_name)) 
                
            if best_auc_sum>.7:
                dice_loss=[]
                otsu_dice_loss=[]
                for current_batch_index in range(0,len(test_id),batch_size):
                    current_batch_id  = [test_id[i] for i in range(current_batch_index,min(len(test_id),current_batch_index+batch_size))]
                    #print(len(current_batch_id))
                    current_batch,current_label=get_data(current_batch_id)
                    #current_batch=current_batch[..., np.newaxis]
                    #current_label=current_label[..., np.newaxis]
                    sess_results = sess.run([cost],feed_dict={X:current_batch,Y:current_label})
                    dice_loss.append(sess_results[0])
                    #print(' Iter: ', iter, " Cost:  %.3f"% sess_results[0],end='\r')
                    sess_results=sess.run(y_pred, feed_dict={X: current_batch})
                    for i in range(len(current_batch_id)):
                        test_example_pred = sess_results[i,:,:,:]
                        #test_example = current_batch[i,:,:,:]
                        test_example_gt = current_label[i,:,:,:] 
                        test_example_pred=np.squeeze(test_example_pred, axis=( 2)) 
                        #test_example =np.squeeze (test_example_gt ,axis=( 2))
                        test_example_gt =np.squeeze (test_example_gt ,axis=( 2))
                        otsu_dice_loss.append(dice(threshold_by_otsu(test_example_pred),test_example_gt))
                print('\n--------Test Loss---------------')
                print(np.mean(dice_loss))
                print('\n--------otsu_dice_loss---------------')
                print(np.mean(otsu_dice_loss))
                print('\n-----------------------') 
        if iter%10==0 and flag==0:
        #if 1:
            #print('\n-----------------------')
            #print('\n--------Regular save model-----')
            #print('\n-----------------------')
            saver = tf.train.Saver()
            model_name = "iter_{}_auc_sum_{:.3}".format(iter, auc_sum)
            saver.save(sess, os.path.join(model_out_dir, model_name)) 