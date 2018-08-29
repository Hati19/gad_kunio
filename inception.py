from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.layers import Input
from keras.utils import np_utils

import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle
from keras.callbacks import ModelCheckpoint

################################################################
out_dir1='/users/PCCF0019/ccf0071/Downloads/gald_seg/data_all/image_seg/'
out_dir2='/users/PCCF0019/ccf0071/Downloads/gald_seg/data_all/image_without_seg/'
img_rows=256
img_cols=256
image_ch=3
##############################################################
def get_data(input_file_name_list):
  
    images=input_file_name_list
    total = len(images) 
    imgs = np.ndarray((total, img_rows, img_cols,image_ch), dtype=np.float32)
    #imgs_mask = np.ndarray((total,img_rows, img_cols), dtype=np.float32)
    i=0
    for image_name in images:
        
        if 'mask'  in image_name:
            continue
        
        image_mask_name = image_name.split('.')[0] + '_mask.npy'
        img = np.load(image_name)
        img=img[:,:,:3]
        #img=np.squeeze(img, axis=0)
        #mean=np.mean(img)
        #print(mean)
       
        #adjusted_stddev = max(np.std(img), 1.0/np.sqrt(img_rows* img_cols*image_ch))
        #print(adjusted_stddev)
        #img=(img - mean) / adjusted_stddev
        #img_mask = np.load(image_mask_name)
        #img_mask=np.squeeze(img_mask, axis=0)
        img = np.array([img])
        #img_mask = np.array([img_mask])
        imgs[i] = img
        #imgs_mask[i] = img_mask
        i+=1
        #if(i%160)
        
    
    #print(imgs.shape)
    #print(imgs_mask.shape)
    
    #return imgs, imgs_mask[..., np.newaxis]
    return imgs
    
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
    
    
###############################################################    
    
# create the base pre-trained model
input_tensor = Input(shape=(256, 256,3))  # this assumes K.image_data_format() == 'channels_last'
base_model = InceptionV3(input_tensor=input_tensor, weights='imagenet', include_top=False)
#base_model = InceptionV3(weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer -- let's say we have 200 classes
predictions = Dense(2, activation='softmax')(x)

# this is the model we will train


model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
model.summary()

# start training data load
total_id, _ =get_train_file_names(out_dir1)
x1=get_data(total_id)
total_id1, _ =get_train_file_names(out_dir2)
x2 =get_data(total_id1)
print(x1.shape)
print(x2.shape)
X=np.concatenate((x1,x2), axis=0)
print(X.shape)
total_mask_id=np.ones(len(total_id))
total_mask_id1=np.zeros(len(total_id1))
print(total_mask_id1)
total_id=total_id+total_id1
total_mask_id=np.concatenate((total_mask_id, total_mask_id1), axis=0)
print(total_mask_id)
print(len(total_id))
print(total_mask_id.shape)
#Y=[1,1,0,0]
encoder = LabelEncoder()
encoder.fit(total_mask_id)
encoded_Y = encoder.transform(total_mask_id)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)
#print(dummy_y)

#Y=[1,1,0,0]
#X=['a','b','c','d']
X,Y=shuffle(X,dummy_y)
#print(Y)
print(X.shape)
#label1=int(X.shape[0]*.7)
label2=int(X.shape[0]*.9)

train_id=X[:label2,:,:,:]
train_mask_id=Y[:label2,:]
#val_id=X[label1:label2,:,:,:]
#val_mask_id=Y[label1:label2,:]
test_id=X[label2:,:,:,:]
test_mask_id=Y[label2:,:]

print(train_id.shape)
print(train_mask_id.shape)
#print(val_id.shape)
#print(val_mask_id.shape)
print(test_id.shape)
print(test_mask_id.shape)
np.save('test_id.npy', test_id)
np.save('test_mask_id.npy', test_mask_id)


mean = np.mean(train_id)  # mean for data centering
std = np.std(train_id)  # std for data normalization

train_id -= mean
train_id /= std


print('-'*30)
print('Creating and compiling model...')
print('-'*30)
#model = get_unet()
model_checkpoint = ModelCheckpoint('weights.h5', monitor='val_loss', save_best_only=True)

print('-'*30)
print('Fitting model...')
print('-'*30)
model.fit(train_id, train_mask_id, batch_size=32, nb_epoch=200, verbose=1, shuffle=True,
          validation_split=0.3,
          callbacks=[model_checkpoint])
          
mean = np.mean(test_id)  # mean for data centering
std = np.std(test_id)  # std for data normalization

test_id -= mean
test_id /= std

print('-'*30)
print('evaluting on test data...')
print('-'*30)
#score = model.evaluate(test_id, test_mask_id, verbose=1)
test_mask_id_pred = model.predict(test_id, verbose=1)
print(test_mask_id_pred)
print(test_mask_id)
print(np.concatenate((total_mask_id, total_mask_id1), axis=1))