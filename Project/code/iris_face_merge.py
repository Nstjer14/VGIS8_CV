#import iris_cnn
#import VGG16_face
import fusion_data_creater
import imageBatchGenerator5_module as batchGen


from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Input, Flatten, Dense, Merge
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.utils import to_categorical
from keras.optimizers import SGD


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

import matplotlib.pyplot as plt
import cv2
from collections import Counter
import numpy as np
import os
import datetime
import random as rd

rd.seed(42)


save_dir = os.path.join(os.getcwd(), 'saved_models')
pic_save_dir = os.path.join(os.getcwd(), 'saved_graphs')
model_name = 'merged_model.h5'
#%% Load data
'''
iris_imageVector = iris_cnn.imageVector
iris_label = iris_cnn.batchLabels_iris

faces_data = VGG16_face.data
faces_label = VGG16_face.label
'''
dataFrame = fusion_data_creater.fusionDataframe

counts = Counter(dataFrame.label)
discardList = []
minNumOfImages = 10
for iris_name,value in counts.items():
    if value<=minNumOfImages:
        discardList.append(iris_name)
dataFrame = dataFrame[~dataFrame['label'].isin(discardList)]
print("Classes' with less than %.f images discarded in total are %.f : " % (minNumOfImages,len(discardList)),discardList)
'''
resized_images_iris = []

temp_for_reshape_iris = dataFrame.iris.values
img_dim_iris = temp_for_reshape_iris[1].shape
imageVector_iris = []

for i in temp_for_reshape_iris:
    imageVector_iris.append(np.array(i.reshape(img_dim_iris)))
imageVector_iris = np.asarray(imageVector_iris)
'''
#dataFrame = dataFrame.reset_index()
resized_images_face = []

temp_for_reshape_face = dataFrame.face.values
img_dim_face = temp_for_reshape_face[1].shape
imageVector_face = []

for i in temp_for_reshape_face:
    imageVector_face.append(np.array(i.reshape(img_dim_face)))
imageVector_face = np.asarray(imageVector_face)


label = dataFrame.label
label = label.tolist() # The list coming from dataFrame is already in the correct format.

#%% Iris preparation
resized_image_iris = []
for image in dataFrame.iris:
    resized_image_iris.append(cv2.resize(image, (64, 64)))

imageVector_face = imageVector_face.astype('float32') # Rescale it from 255 to 0-1.
imageVector_face = imageVector_face/255.

batchImages_iris = []
batchImages_face = []
batchLabels = []

for i in range(0,len(resized_image_iris)):
    batchesInTupples_iris = batchGen.imageBatchGenerator5(resized_image_iris[i])
    for j in batchesInTupples_iris:
        #plt.imshow(j, cmap='gray')
        batchImages_iris.append(j)
        #batchLabels_iris.append(label[i])
        batchImages_iris.append(cv2.flip(j,0)) # adding horisontal flip
        #batchLabels_iris.append(label[i])
        
    batchesInTupples_face = batchGen.imageBatchGenerator5(imageVector_face[i])
    for j in batchesInTupples_face:
        #plt.imshow(j, cmap='gray')
        batchImages_face.append(j)
        #batchLabels_iris.append(label[i])
        batchImages_face.append(cv2.flip(j,0)) # adding horisontal flip
        #batchLabels_iris.append(label[i])
    for k in range(0,len(batchesInTupples_face)):
        batchLabels.append(label[i])
        batchLabels.append(label[i])
        
reshapeDims = batchImages_iris[1].shape
imageVector_iris = np.asarray(batchImages_iris)
imageVector_iris = imageVector_iris.reshape(imageVector_iris.shape[0],1,58,58) # format it from (64,512) to (64,512,1) since it is an image with only 1 channel
#imageVector = imageVector.reshape(imageVector.shape[0],1,64,512) # format it from (64,512) to (64,512,1) since it is an image with only 1 channel
imageVector_iris = imageVector_iris.astype('float32') # Rescale it from 255 to 0-1.
imageVector_iris = imageVector_iris/255.
imageVector_face = np.asarray(batchImages_face)

#label_one_hot_encoded = to_categorical(batchLabels)

# integer encode
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(batchLabels)
print(integer_encoded)

# binary encode
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
label_onehot = onehot_encoded

label_one_hot_encoded = label_onehot

print('Iris Training data shape after reshape : ', imageVector_iris.shape)
print('Face Training data shape after reshape : ', imageVector_face.shape)
print('Label Training data shape after reshape : ', label_one_hot_encoded.shape)

#label = batchLabels_iris # set labels to be the batchLabels_iris. This is a quick and dirty
uniqueClasses=np.unique(batchLabels)
NuniqueClasses=len(uniqueClasses)

#%% Iris CNN
input_shape = imageVector_iris[1].shape
num_classes = NuniqueClasses
batch_size = 64
epochs = 50

iris_model = Sequential()
iris_model.add(Conv2D(6, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape,
                 data_format='channels_first',
                 padding = "same"))
iris_model.add(MaxPooling2D(pool_size=(2, 2),data_format='channels_first'))
#iris_model.add(Dropout(0.25))
iris_model.add(Conv2D(32,
                 kernel_size=(5, 5),
                 activation='relu',
                 padding = "same",
                 data_format='channels_first'))
iris_model.add(MaxPooling2D(pool_size=(2, 2),data_format='channels_first'))

iris_model.add(Conv2D(64,
                 kernel_size=(5, 5),
                 activation='relu',
                 padding = "same",
                 data_format='channels_first'))
iris_model.add(MaxPooling2D(pool_size=(2, 2),data_format='channels_first'))

iris_model.add(Conv2D(256,
                 kernel_size=(5, 5),
                 activation='relu',
                 padding = "same",
                 data_format='channels_first'))

iris_model.add(Flatten())
iris_model.add(Dense(1024, activation='relu'))
iris_model.add(Dropout(0.5))
iris_model.add(Dense(1024, activation='relu'))
#iris_model.add(Dropout(0.5))
#iris_model.add(Dense(num_classes, activation='softmax',))

#%% Face CNN
model_vgg16_conv = VGG16(weights='imagenet', include_top=False)
model_vgg16_conv.summary()

face_input_shape = imageVector_face[1].shape 
    #Create your own input format (here 3x200x200)
input_vgg16 = Input(shape=face_input_shape,name = 'image_input')
    
    #Use the generated model 
output_vgg16_conv = model_vgg16_conv(input_vgg16)
    
    #Add the fully-connected layers 
x = Flatten(name='flatten')(output_vgg16_conv)
x = Dense(4096, activation='relu', name='fc1')(x)
x = Dropout(0.5)(x)
x = Dense(4096, activation='relu', name='fc2')(x)
#x = Dropout(0.5)(x)
#x = Dense(num_classes, activation='softmax', name='predictions')(x)


#Create your own model 
face_model = Model(input=input_vgg16, output=x)

iris_model.summary()
face_model.summary()

merged_model = Sequential()

merged_model.add(Merge([iris_model, face_model], mode = 'concat'))
merged_model.add(Dense(1024, activation='relu'))
merged_model.add(Dropout(0.5))
merged_model.add(Dense(1024, activation='relu'))
merged_model.add(Dropout(0.5))
merged_model.add(Dense(num_classes,init = 'normal', activation='softmax',))
print(merged_model.summary())

sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
merged_model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
history = merged_model.fit([imageVector_iris, imageVector_face],
                 label_one_hot_encoded,
                 batch_size=batch_size,
                 epochs=epochs,
                 verbose=1,
                 shuffle=True,
                 validation_split=0.2)


acc_round ='test'# str(round(score[1]*100,2))
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
namestamp = timestamp + 'acc_' + acc_round 
model_name = namestamp + '_VGG16_face.h5'

if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
merged_model.save(model_path)
print('Saved trained model at %s ' % model_path)
    
    
    #%%
print(history)
fig1, ax_acc = plt.subplots()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Model - Accuracy')
plt.legend(['Training', 'Validation'], loc='lower right')

if not os.path.isdir(pic_save_dir):
    os.makedirs(pic_save_dir)
pic_path = os.path.abspath(pic_save_dir)
plt.savefig(pic_path + '/'+model_name+'acc.pdf')
print('Saved graphs at %s ' % pic_path)

plt.show()

# Loss
fig2, ax_loss = plt.subplots()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Model- Loss')
plt.legend(['Training', 'Validation'], loc='upper right')
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training', 'Validation'], loc='lower right')

if not os.path.isdir(pic_save_dir):
    os.makedirs(pic_save_dir)
pic_path = os.path.abspath(pic_save_dir)
print(pic_path)
plt.savefig(pic_path + '/'+model_name+'loss.pdf')
print('Saved graphs at %s ' % pic_path)

plt.show()
    
                     
                 
#                 label_one_hot_encoded, batch_size = 64, nb_epoch = 100, verbose = 1)