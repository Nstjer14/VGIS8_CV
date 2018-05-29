'''Trains a simple convnet on the MNIST dataset.

Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import os
import random as rd
from keras.preprocessing.image import ImageDataGenerator
import imagePatchGenerator5_module as patchGen
import iris_face_merge_cnn_data_splitter as getData
from sklearn.utils import shuffle
import datetime
rd.seed(42)

#%%
import matplotlib.pyplot as plt
import load_images_2_python
import pandas as pd
from collections import Counter
import numpy as np
from sklearn.model_selection import train_test_split

from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

import cv2

save_dir = os.path.join(os.getcwd(), 'saved_models')
pic_save_dir = os.path.join(os.getcwd(), 'saved_graphs')
dataFrame = pd.read_pickle("pythonDatabase")
dataFrame = shuffle(dataFrame)

'''
# Comment this part in to use the data before the merger
# Dropping classes with less than 10 images
counts = Counter(dataFrame.label)
discardList = []
minNumOfImages = 10
for iris_name,value in counts.items():
    if value<=minNumOfImages:
        discardList.append(iris_name)
dataFrame = dataFrame[~dataFrame['label'].isin(discardList)]
print("Classes' with less than %.f images discarded in total are %.f : " % (minNumOfImages,len(discardList)),discardList)


resized_images = []

temp_for_reshape = dataFrame.image.values
img_dim = temp_for_reshape[1].shape
imageVector = []

for i in temp_for_reshape:
    imageVector.append(np.array(i.reshape(img_dim)))
imageVector = np.asarray(imageVector)

label = dataFrame.label
label = label.tolist() # The list coming from dataFrame is already in the correct format.

#%% Explore data
print('Training data shape : ', imageVector.shape)
uniqueClasses=np.unique(label)
NuniqueClasses=len(uniqueClasses)
print("Number of classes: ",NuniqueClasses)

plt.figure(figsize=[8,64])

plt.subplot(121)
plt.imshow(imageVector[0,:,:], cmap='gray')
plt.title("Ground Truth : {}".format(label[0]))




#%% Preprocess for cnn
resized_image = []
for image in dataFrame.image:
    resized_image.append(cv2.resize(image, (64, 64)))

batchImages = []
batchLabels = []
for i in range(0,len(resized_image)):
    batchesInTupples = patchGen.imagePatchGenerator5(resized_image[i])
    for j in batchesInTupples:
        #plt.imshow(j, cmap='gray')
        batchImages.append(j)
        batchLabels.append(label[i])
        batchImages.append(cv2.flip(j,1)) # adding horisontal flip
        batchLabels.append(label[i])
        
reshapeDims = batchImages[1].shape
imageVector = np.asarray(batchImages)
imageVector = imageVector.reshape(imageVector.shape[0],1,58,58) # format it from (64,512) to (64,512,1) since it is an image with only 1 channel
#imageVector = imageVector.reshape(imageVector.shape[0],1,64,512) # format it from (64,512) to (64,512,1) since it is an image with only 1 channel
imageVector = imageVector.astype('float32') # Rescale it from 255 to 0-1.
imageVector = imageVector/255.
print('Training data shape after reshape : ', imageVector.shape)
label = batchLabels # set labels to be the batchLabels. This is a quick and dirty


# integer encode
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(label)
print(integer_encoded)

# binary encode
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
label_onehot = onehot_encoded
print("Normalised pixels to be 0-1 and one hot encoding done:",label_onehot.shape)

#train_X,valid_X,train_label,valid_label = train_test_split(imageVector, label_onehot, test_size=0.2, random_state=13)
'''
# Comment this section out when using the data before the merger for fusion dataset
train_X =  getData.train_iris_X
train_label =  getData.train_label

test_iris_X =  getData.test_iris_X
test_label =  getData.test_label

valid_X =  getData.validation_iris_X
valid_label =  getData.validation_label

NuniqueClasses = len(np.unique(getData.label))

#%%

x_train = train_X
x_test = valid_X#test_iris_X # 
y_train = train_label
y_test = valid_label#test_label##

batch_size = 128
epochs = 50

print("Shape of x_train",x_train.shape)
print("Shape of y_train",y_train.shape)
print("Shape of x_test",x_test.shape)
print("Shape of y_test",y_test.shape)


input_shape = train_X[0].shape
num_classes = NuniqueClasses

from keras.models import Model
from keras.layers import Input

iris_model_in = Input(shape = input_shape)
conv1 = Conv2D(6, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape,
                 data_format='channels_first',
                 padding = "same")(iris_model_in)

max1 = MaxPooling2D(pool_size=(2, 2),data_format='channels_first')(conv1)

conv2 = Conv2D(32,
                 kernel_size=(5, 5),
                 activation='relu',
                 padding = "same",
                 data_format='channels_first')(max1)

max2 = MaxPooling2D(pool_size=(2, 2),data_format='channels_first')(conv2)

conv3 = Conv2D(64,
                 kernel_size=(5, 5),
                 activation='relu',
                 padding = "same",
                 data_format='channels_first')(max2)

max3 = MaxPooling2D(pool_size=(2, 2),data_format='channels_first')(conv3)

conv4 = Conv2D(256,
                 kernel_size=(5, 5),
                 activation='relu',
                 padding = "same",
                 data_format='channels_first')(max3)


flat = Flatten()(conv4)
dense1 = Dense(1024, activation='relu')(flat)
drop1 = Dropout(0.5)(dense1)
dense2 = Dense(1024, activation='relu')(drop1)
drop2 = Dropout(0.5)(dense2)

iris_out = Dense(num_classes, activation='softmax')(drop2)
model = Model(iris_model_in,iris_out)

learningrate = 1e-3
adagrad = keras.optimizers.Adagrad(lr=learningrate, epsilon=None, decay=0.0005)
#model.compile(loss='categorical_crossentropy', optimizer=sgd)
model.compile(loss='categorical_crossentropy',
              optimizer=adagrad,
              metrics=['accuracy'])
model.summary()

history = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          shuffle=True,
          verbose=1,
          validation_data=(valid_X,valid_label)) 
"""
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)
"""

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1]*100)

acc_round = str(round(score[1]*100,2))
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
namestamp = timestamp + 'acc_' + acc_round 
model_name = namestamp + '_iris_cnn_test.h5'

if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
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
plt.savefig(pic_path + '/'+namestamp+'acc.pdf')
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
plt.savefig(pic_path + '/'+namestamp+'loss.pdf')
print('Saved graphs at %s ' % pic_path)

plt.show()

