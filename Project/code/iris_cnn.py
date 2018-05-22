# -*- coding: utf-8 -*-
"""
Created on Thu May 10 17:41:04 2018

@author: Shaggy
"""
import tensorflow as tf
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))


import matplotlib.pyplot as plt
import load_images_2_python
import imageBatchGenerator5_module as batchGen
import pandas as pd
from collections import Counter
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator

from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

import cv2
#%% Load data
dataFrame = pd.read_pickle("pythonDatabase")

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
    batchesInTupples = batchGen.imageBatchGenerator5(resized_image[i])
    for j in batchesInTupples:
        #plt.imshow(j, cmap='gray')
        batchImages.append(j)
        batchLabels.append(label[i])
        batchImages.append(cv2.flip(j,0)) # adding horisontal flip
        batchLabels.append(label[i])
        
reshapeDims = batchImages[1].shape
imageVector = np.asarray(batchImages)
imageVector = imageVector.reshape(imageVector.shape[0],1,58,58) # format it from (64,512) to (64,512,1) since it is an image with only 1 channel
#imageVector = imageVector.reshape(-1, 64,512, 1) # format it from (64,512) to (64,512,1) since it is an image with only 1 channel
imageVector = imageVector.astype('float32') # Rescale it from 255 to 0-1.
imageVector = imageVector/255.

#datagen = ImageDataGenerator(horizontal_flip=True)
#a,y = datagen.flow(imageVector,batchLabels)
print('Training data shape after reshape : ', imageVector.shape)

'''
resized_image2 = []
for image in dataFrame.image:
    resized_image2.append(cv2.resize(image, (64, 64)))
imageVector2 = np.asarray(resized_image2)
imageVector2 = imageVector.reshape(imageVector.shape[0],1,64,64) # format it from (64,512) to (64,512,1) since it is an image with only 1 channel
#imageVector = imageVector.reshape(imageVector.shape[0],1,64,512) # format it from (64,512) to (64,512,1) since it is an image with only 1 channel
imageVector = imageVector.astype('float32') # Rescale it from 255 to 0-1.
imageVector = imageVector/255.
print('Training data shape after reshape : ', imageVector.shape)
''' 
    
label = batchLabels
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

train_X,valid_X,train_label,valid_label = train_test_split(imageVector, label_onehot, test_size=0.2, random_state=13)


#%% Load MNIST for testing
'''
from keras.datasets import mnist
 
# Load pre-shuffled MNIST data into train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print (X_train.shape)
plt.imshow(X_train[0])
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
from keras.utils import np_utils
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

train_X = X_train
valid_X = y_train
train_label = X_test
valid_label =y_test

print(train_X.shape)
print(valid_X.shape)
'''
#%% Make the CNN structure 

import keras
from keras.models import Sequential,Input,Model
from keras.layers import Activation,Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras import optimizers
from keras import regularizers
#from keras import backend as K

# CNN architecture
img_shape = train_X[0].shape
num_classes = NuniqueClasses
#K.set_image_dim_ordering('th')

'''
# CNN for MNIST testing
num_classes = 10
model = Sequential()
model.add(Conv2D(32, 3, 3, activation='relu', input_shape=(1,28,28)))
model.add(Conv2D(32, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2),padding='same'))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
'''
model = Sequential()
model.add(Conv2D(6, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=img_shape,
                 data_format='channels_first',
                 padding = "same"))
model.add(MaxPooling2D(pool_size=(2, 2),data_format='channels_first'))
#model.add(Dropout(0.25))
model.add(Conv2D(32,
                 kernel_size=(5, 5),
                 activation='relu',
                 padding = "same",
                 data_format='channels_first'))
model.add(MaxPooling2D(pool_size=(2, 2),data_format='channels_first'))

model.add(Conv2D(64,
                 kernel_size=(5, 5),
                 activation='relu',
                 padding = "same",
                 data_format='channels_first'))
model.add(MaxPooling2D(pool_size=(2, 2),data_format='channels_first'))

model.add(Conv2D(256,
                 kernel_size=(5, 5),
                 activation='relu',
                 padding = "same",
                 data_format='channels_first'))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax',))

model.summary()
#print(model.get_config())	
#Parameters
batch_size = 150
epochs = 100
learningrate = 0.03#1e-3

adagrad = keras.optimizers.Adagrad(lr=learningrate, epsilon=None, decay=0.0000)
#model.compile(loss='categorical_crossentropy', optimizer=sgd)

model.compile(loss='categorical_crossentropy',
              optimizer=adagrad,
              metrics=['accuracy'])

history = model.fit(train_X, train_label, 
          batch_size=batch_size, nb_epoch=epochs, verbose=1)

score = model.evaluate(valid_X, valid_label, verbose=0)
print("Accuracy: %.2f%%" % (score[1]*100))

#%%
# Accuracy
print(history)
fig1, ax_acc = plt.subplots()
plt.plot(history.history['acc'])
#   plt.plot(history.history['val_acc'])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Model - Accuracy')
plt.legend(['Training', 'Validation'], loc='lower right')
plt.show()

# Loss
fig2, ax_loss = plt.subplots()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Model- Loss')
plt.legend(['Training', 'Validation'], loc='upper right')
plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
plt.show()
#%%
'''
from keras.datasets import fashion_mnist
(train_X,train_Y), (test_X,test_Y) = fashion_mnist.load_data()

import numpy as np
from keras.utils import to_categorical
import matplotlib.pyplot as plt
#%matplotlib inline

print('Training data shape : ', train_X.shape, train_Y.shape)

print('Testing data shape : ', test_X.shape, test_Y.shape)


classes = np.unique(train_Y)
nClasses = len(classes)
print('Total number of outputs : ', nClasses)
print('Output classes : ', classes)

plt.figure(figsize=[5,5])

# Display the first image in training data
plt.subplot(121)
plt.imshow(train_X[0,:,:], cmap='gray')
plt.title("Ground Truth : {}".format(train_Y[0]))

# Display the first image in testing data
plt.subplot(122)
plt.imshow(test_X[0,:,:], cmap='gray')
plt.title("Ground Truth : {}".format(test_Y[0]))

train_X = train_X.reshape(-1, 28,28, 1)
test_X = test_X.reshape(-1, 28,28, 1)
train_X.shape, test_X.shape

train_X = train_X.astype('float32')
test_X = test_X.astype('float32')
train_X = train_X / 255.
test_X = test_X / 255.


# Change the labels from categorical to one-hot encoding
train_Y_one_hot = to_categorical(train_Y)
test_Y_one_hot = to_categorical(test_Y)

# Display the change for category label using one-hot encoding
print('Original label:', train_Y[0])
print('After conversion to one-hot:', train_Y_one_hot[0])

from sklearn.model_selection import train_test_split
train_X,valid_X,train_label,valid_label = train_test_split(train_X, train_Y_one_hot, test_size=0.2, random_state=13)

train_X.shape,valid_X.shape,train_label.shape,valid_label.shape


import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU

batch_size = 64
epochs = 20
num_classes = 10

fashion_model = Sequential()
fashion_model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',input_shape=(28,28,1),padding='same'))
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(MaxPooling2D((2, 2),padding='same'))
fashion_model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
fashion_model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
fashion_model.add(LeakyReLU(alpha=0.1))                  
fashion_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
fashion_model.add(Flatten())
fashion_model.add(Dense(128, activation='linear'))
fashion_model.add(LeakyReLU(alpha=0.1))                  
fashion_model.add(Dense(num_classes, activation='softmax'))

fashion_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])

fashion_model.summary()

fashion_train = fashion_model.fit(train_X, train_label, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(valid_X, valid_label))
'''