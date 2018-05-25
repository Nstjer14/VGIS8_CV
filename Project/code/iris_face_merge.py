#import iris_cnn
#import VGG16_face
import fusion_data_creater
import imagePatchGenerator5_module as patchGen
import iris_face_merge_cnn_data_splitter as getData

import keras
from keras.models import Sequential
from keras.layers.merge import Concatenate
from keras.layers import Dense, Dropout, Flatten, Input, Flatten, Dense, Merge
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.utils import to_categorical
from keras.optimizers import SGD


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

import pandas as pd
import matplotlib.pyplot as plt
import cv2
from collections import Counter
import numpy as np
import os
import datetime
import random as rd
from sklearn.utils import shuffle

rd.seed(42)


save_dir = os.path.join(os.getcwd(), 'saved_models')
pic_save_dir = os.path.join(os.getcwd(), 'saved_graphs')

#%% Load data
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

#%% Iris CNN
input_shape = train_iris_X[1].shape
num_classes = NuniqueClasses
batch_size = 68
epochs = 150

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
drop1_iris = Dropout(0.5)(dense1)
dense2 = Dense(1024, activation='relu')(drop1_iris)

iris_out = dense2# Dense(num_classes, activation='softmax')(flat)
iris_model = Model(iris_model_in,output=iris_out)

''' # Other syntax for creating the same model
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
'''
#%% Face CNN

model_vgg16_conv = VGG16(weights='imagenet', include_top=False)
model_vgg16_conv.summary()

face_input_shape = train_face_X[1].shape 
    #Create your own input format (here 3x200x200)
input_vgg16 = Input(shape=face_input_shape,name = 'image_input')
    
    #Use the generated model 
output_vgg16_conv = model_vgg16_conv(input_vgg16)
#output_vgg16_conv =  Conv2D(6, kernel_size=(3, 3),
#                 activation='relu',
#                 input_shape=input_shape,
#                 data_format='channels_last',
#                 padding = "same")(input_vgg16)   
    #Add the fully-connected layers 
flat_face = Flatten(name='flatten')(output_vgg16_conv)
fc1_face = Dense(4096, activation='relu', name='fc1')(flat_face)
drop1_face = Dropout(0.5)(fc1_face)
fc2_face = Dense(4096, activation='relu', name='fc2')(drop1_face)
#x = Dropout(0.5)(x)
face_out = fc2_face#Dense(num_classes, activation='softmax', name='predictions')(flat_face)


#Create your own model 
face_model = Model(input=input_vgg16, output=face_out)

iris_model.summary()
#face_model.summary()
''' # Other syneax for creating same model
input_shape_face = imageVector_face[1].shape
#num_classes = NuniqueClasses
#batch_size = 128
#epochs = 50

face_model = Sequential()
face_model.add(Conv2D(6, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape_face,
                 data_format='channels_last',
                 padding = "same"))
face_model.add(MaxPooling2D(pool_size=(2, 2),data_format='channels_last'))
#iris_model.add(Dropout(0.25))
face_model.add(Conv2D(32,
                 kernel_size=(5, 5),
                 activation='relu',
                 padding = "same",
                 data_format='channels_last'))
face_model.add(MaxPooling2D(pool_size=(2, 2),data_format='channels_last'))

face_model.add(Conv2D(64,
                 kernel_size=(5, 5),
                 activation='relu',
                 padding = "same",
                 data_format='channels_last'))
face_model.add(MaxPooling2D(pool_size=(2, 2),data_format='channels_last'))

face_model.add(Conv2D(256,
                 kernel_size=(5, 5),
                 activation='relu',
                 padding = "same",
                 data_format='channels_last'))

face_model.add(Flatten())
face_model.add(Dense(1024, activation='relu'))
face_model.add(Dropout(0.5))
face_model.add(Dense(1024, activation='relu'))
'''
#%% Merged CNN

#merged_model = Sequential()

merged_layer = Concatenate()([iris_out, face_out])
fc1_merged = Dense((1024+4096), activation='relu',name = 'merged_fc1')(merged_layer)
drop1 = Dropout(0.5)(fc1_merged)
fc2_merged = Dense((1024+4096), activation='relu',name = 'merged_fc2')(drop1)
drop2 = Dropout(0.5)(fc2_merged)

#merged_model.add(Dense(1024, activation='relu',))
output_merged = Dense(num_classes, activation='softmax')(drop2)


merged_model = Model(inputs=[iris_model_in, input_vgg16], outputs=output_merged)

print(merged_model.summary())
'''
merged_model.add(Merge([iris_model, face_model], mode = 'concat'))
#merged_model.add(Dense(1024, activation='relu',))
#merged_model.add(Dropout(0.5))
#merged_model.add(Dense(1024, activation='relu'))
#merged_model.add(Dropout(0.5))
merged_model.add(Dense(num_classes, activation='sigmoid',))
'''
#print(merged_model.summary())


learningrate = (1e-3)
adagrad = keras.optimizers.Adagrad(lr=learningrate, epsilon=None, decay=0.0005)

#sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
#keras.optimizers.Nadam()
merged_model.compile(adagrad, loss='categorical_crossentropy', metrics=['accuracy'])


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
    
                     
                 
#                 label_one_hot_encoded, batch_size = 64, nb_epoch = 100, verbose = 1)
