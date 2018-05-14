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

batch_size = 128
num_classes = 10
epochs = 2

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


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
#for image in dataFrame.image:
#    resized_image.append(cv2.resize(image, (64, 64)))
#imageVector = np.asarray(resized_image)
#imageVector = imageVector.reshape(-1, 64,64, 1) # format it from (64,512) to (64,512,1) since it is an image with only 1 channel
imageVector = imageVector.reshape(imageVector.shape[0],1,64,512) # format it from (64,512) to (64,512,1) since it is an image with only 1 channel
imageVector = imageVector.astype('float32') # Rescale it from 255 to 0-1.
imageVector = imageVector/255.
print('Training data shape after reshape : ', imageVector.shape)



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


#%%

x_train = train_X
x_test = valid_X
y_train = train_label
y_test = valid_label

input_shape = imageVector[0].shape
num_classes = NuniqueClasses

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
