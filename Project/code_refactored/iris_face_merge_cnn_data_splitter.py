#import iris_cnn
#import VGG16_face
import fusion_data_creater
import imagePatchGenerator5_module as patchGen

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

import cv2
from collections import Counter
import numpy as np
import random as rd
from sklearn.utils import shuffle

rd.seed(42)

##%
def prepFunction(genericDataframe,genericLabel):
    dataFrame = genericDataframe
    label = genericLabel
    resized_images_face = []
    
    temp_for_reshape_face = dataFrame.face.values
    img_dim_face = temp_for_reshape_face[1].shape
    imageVector_face = []
    
    for i in temp_for_reshape_face:
        imageVector_face.append(np.array(i.reshape(img_dim_face)))
    imageVector_face = np.asarray(imageVector_face)
    
    resized_image_iris = []
    for image in dataFrame.iris:
        resized_image_iris.append(cv2.resize(image, (64, 64)))
    
    
    
    patchImages_iris = []
    patchImages_face = []
    patchLabels = []
    
    for i in range(0,len(resized_image_iris)):
        batchesInTupples_iris = patchGen.imagePatchGenerator5(resized_image_iris[i])
        for j in batchesInTupples_iris:
            #plt.imshow(j, cmap='gray')
            patchImages_iris.append(j)
            #patchLabels_iris.append(label[i])
            patchImages_iris.append(cv2.flip(j,1)) # adding horisontal flip
            #patchLabels_iris.append(label[i])
            
        batchesInTupples_face = patchGen.imagePatchGenerator5(imageVector_face[i])
        for j in batchesInTupples_face:
            #plt.imshow(j, cmap='gray')
            patchImages_face.append(j)
            #patchLabels_iris.append(label[i])
            patchImages_face.append(cv2.flip(j,1)) # adding horisontal flip
            #patchLabels_iris.append(label[i])
        for k in range(0,len(batchesInTupples_face)):
            patchLabels.append(label[i])
            patchLabels.append(label[i])
    
    #patchImages_iris = resized_image_iris # Comment in to use original images and not batches
    #patchImages_face = imageVector_face # Comment in to use original images and not batches
    #patchLabels = label        # Comment in to use original images and not batches
    reshapeDims = patchImages_iris[1].shape
    imageVector_iris = np.asarray(patchImages_iris)
    imageVector_iris = imageVector_iris.reshape(imageVector_iris.shape[0],1,reshapeDims[0],reshapeDims[1]) # format it from (64,512) to (64,512,1) since it is an image with only 1 channel
    #imageVector = imageVector.reshape(imageVector.shape[0],1,64,512) # format it from (64,512) to (64,512,1) since it is an image with only 1 channel
    imageVector_iris = imageVector_iris.astype('float32') # Rescale it from 255 to 0-1.
    imageVector_iris = imageVector_iris/255.
    
    
    imageVector_face = np.asarray(patchImages_face)
    imageVector_face = imageVector_face.astype('float32') # Rescale it from 255 to 0-1.
    imageVector_face = imageVector_face/255.
    #label_one_hot_encoded = to_categorical(patchLabels)
    
    # integer encode
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(patchLabels)
    #print(integer_encoded)
    
    # binary encode
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    label_onehot = onehot_encoded
    
    label_one_hot_encoded = label_onehot
    return imageVector_iris,imageVector_face,label_one_hot_encoded

#%% Load data

dataFrame = fusion_data_creater.fusionDataframe
dataFrame = shuffle(dataFrame)
'''
counts = Counter(dataFrame.label)
discardList = []
minNumOfImages = 10
for iris_name,value in counts.items():
    if value<=minNumOfImages:
        discardList.append(iris_name)
dataFrame = dataFrame[~dataFrame['label'].isin(discardList)]
print("Classes' with less than %.f images discarded in total are %.f : " % (minNumOfImages,len(discardList)),discardList)
'''
label = dataFrame.label
label = label.tolist() # The list coming from dataFrame is already in the correct format.
train_dataFrame_X,temp_X,train_dataFrame_y,temp_y = train_test_split(dataFrame,label,stratify = label,test_size=0.30)
#print(len(np.unique(train_dataFrame_y)))

test_dataFrame_X,validation_dataFrame_X,test_dataFrame_y,validation_dataFrame_y = train_test_split(temp_X,temp_y,stratify =temp_y, test_size=0.50)

#bUniq=np.unique(validation_dataFrame_y)
#aUniq=np.unique(test_dataFrame_y)

#print(len(bUniq),len(aUniq))
train_iris_X,train_face_X,train_label = prepFunction(train_dataFrame_X,train_dataFrame_y)
validation_iris_X,validation_face_X,validation_label = prepFunction(validation_dataFrame_X,validation_dataFrame_y)
test_iris_X,test_face_X,test_label = prepFunction(test_dataFrame_X,test_dataFrame_y)

#len(np.unique(dataFrame.label))

#%%

