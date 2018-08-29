import cv2
import numpy as np
from skimage.io import imsave, imread
from nibabel.testing import data_path
import nibabel as nib
import random







#################################################
read_txt='/users/PCCF0019/ccf0071/Downloads/gald_seg/vol.txt'
location='/users/PCCF0019/ccf0064/deep_learning_segment_gad/'
out_dir1='/users/PCCF0019/ccf0071/Downloads/gald_seg/data_all/image_seg/'
out_dir2='/users/PCCF0019/ccf0071/Downloads/gald_seg/data_all/image_without_seg/'

###################################################
count=0

imgs_id=[]
with open(read_txt, 'r') as filehandle:  
    for line in filehandle:
        count+=1
        # remove linebreak which is the last character of the string
        currentPlace = line[:-1]
        #print(list(filter(None, currentPlace.split(' '))))
        currentPlace=list(filter(None, currentPlace.split(' ')))
        if (float(currentPlace[0]) )> 0:
            #print(currentPlace[-1])
            imgs_id.append(currentPlace[-1])
print(len(imgs_id))
print(count)
count=0
count1=0
for image_id in imgs_id:
    y=np.array(nib.load(image_id).dataobj)
    name=image_id.split('/')[-1]
    name1='_'.join(name.split('_')[0:4])
    name2 = location+name1+'_IM5.nii.gz'
    
    x=np.array(nib.load(name2).dataobj)
    print(name2)
    #print(y.shape)
    for i in range(y.shape[2]):
        img=x[:,:,i,:]
        mask=y[:,:,i]
        #print('id: %d Sum: %d'%(i, np.sum(mask)))
        
        if (np.sum(mask)>0):
            #print(np.amax(img))
            #print(img.shape)
            #print(mask.shape)
            print('slice:%d npsum: %d '%(i,np.sum(mask)))
            #np.save(out_dir1+ name1 + '_'+ "%03d"%i+'.npy', img)
            
            
            count+=1
        else:          #uncomment it for data_all
            if random.random() < 0.2 and np.sum(mask)==0:
                #np.save(out_dir2+ name1 + '_'+ "%03d"%i+'.npy', img)
                count1+=1
print(count)  
print(count1)
print(count/(count+count1))
    #print(np.amax(x))
    #print(np.amax(y))
    
    #name1=name.split('_')[0]
'''for image_id in imgs_id:
    y=np.array(nib.load(image_id).dataobj)
    name=image_id.split('/')[-1]
    name1='_'.join(name.split('_')[0:4])
    name2 = location+name1+'_IM5.nii.gz'
    
    x=np.array(nib.load(name2).dataobj)
    print(name2)
    #print(y.shape)
    for i in range(y.shape[2]):
        img=x[:,:,i,:]
        mask=y[:,:,i]
        #print('id: %d Sum: %d'%(i, np.sum(mask)))
        
        if (np.sum(mask)>10):
            #print(np.amax(img))
            #print(img.shape)
            #print(mask.shape)
            #print(np.sum(img))
            np.save(out_dir1+ name1 + '_'+ "%03d"%i+'.npy', img)
            #np.save(out_dir1+ name1 + '_'+ "%03d"%i+'_mask.npy', mask)
            
            count+=1
        else:          #uncomment it for data_all
            if random.random() < 0.2 and np.sum(mask)==0:
                np.save(out_dir2+ name1 + '_'+ "%03d"%i+'.npy', img)
                #np.save(out_dir2+ name1 + '_'+ "%03d"%i+'_mask.npy', mask)
                count1+=1
print(count)  
print(count1)
print(count/(count+count1))
    #print(np.amax(x))
    #print(np.amax(y))
    
    #name1=name.split('_')[0]'''