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
from keras.applications.vgg16 import VGG16
from keras.layers.merge import Concatenate
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

train_iris_X =  getData.train_iris_X
train_face_X =  getData.train_face_X
train_label =  getData.train_label

test_iris_X =  getData.test_iris_X
test_face_X =  getData.test_face_X
test_label =  getData.test_label

validation_iris_X =  getData.validation_iris_X
validation_face_X =  getData.validation_face_X
validation_label =  getData.validation_label

NuniqueClasses = len(np.unique(getData.label))

#%% The iris CNN

#x_train = train_X
#x_test = test_iris_X #valid_X
#y_train = train_label
#y_test = test_label#valid_label#

batch_size = 128
epochs = 150

input_shape = train_iris_X[0].shape
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
iris_out = dense2
iris_model = Model(iris_model_in,iris_out)

#%% The face CNN

model_vgg16_conv = VGG16(weights='imagenet', include_top=False)
model_vgg16_conv.summary()

face_input_shape = train_face_X[1].shape 
input_vgg16 = Input(shape=face_input_shape,name = 'image_input')
    
output_vgg16_conv = model_vgg16_conv(input_vgg16)

flat_face = Flatten(name='flatten')(output_vgg16_conv)
fc1_face = Dense(1024, activation='relu', name='fc1')(flat_face)
drop1_face = Dropout(0.5)(fc1_face)
fc2_face = Dense(1024, activation='relu', name='fc2')(drop1_face)
face_out = fc2_face

face_model = Model(input=input_vgg16, output=face_out)


#%% Merged CNN

merged_layer = Concatenate()([iris_out, face_out])
fc1_merged = Dense((1024*2), activation='relu',name = 'merged_fc1')(merged_layer)
drop1 = Dropout(0.5)(fc1_merged)
fc2_merged = Dense((1024*2), activation='relu',name = 'merged_fc2')(drop1)
drop2 = Dropout(0.5)(fc2_merged)

output_merged = Dense(num_classes, activation='softmax')(drop2)


merged_model = Model(inputs=[iris_model_in, input_vgg16], outputs=output_merged)

print(merged_model.summary())



#%%

learningrate = 1e-3
adagrad = keras.optimizers.Adagrad(lr=learningrate, epsilon=None, decay=0.0005)
merged_model.compile(loss='categorical_crossentropy',
              optimizer=adagrad,
              metrics=['accuracy'])
merged_model.summary()

history = merged_model.fit([train_iris_X, train_face_X],
                 train_label,
                 batch_size=batch_size,
                 epochs=epochs,
                 verbose=1,
                 shuffle=True,
                 validation_data = ([validation_iris_X, validation_face_X],validation_label))

score = merged_model.evaluate([test_iris_X, test_face_X], test_label, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1]*100)
#%%Saving models and graphs
acc_round = str(round(score[1]*100,2))
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
namestamp = timestamp + 'acc_' + acc_round 
model_name = namestamp + '_merged_iris_face.h5'

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