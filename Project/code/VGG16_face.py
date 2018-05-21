# VGG16 face recognition test


from __future__ import print_function
import numpy as np
import os
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import matplotlib.pyplot as plt
#from datasets import dataset_utils
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
import random as rd
from keras.utils import to_categorical
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.layers import Input, Flatten, Dense
from keras.models import Model




rd.seed(42)

save_dir = os.path.join(os.getcwd(), 'saved_models')
pic_save_dir = os.path.join(os.getcwd(), 'saved_graphs')
model_name = 'face_cnn_test.h5'

# load images for individuals w/ 10+ images and produce centered 64x64 images from orig. 250x250 images
lfw_people = fetch_lfw_people(min_faces_per_person=10, 
                              slice_ = (slice(61,189),slice(61,189)),
                              resize=0.5, color = True)

#for name in lfw_people.target_names:
#    print(name)

# access the images
X = lfw_people.images
#print(X[1].shape)


# access the class labels
y = lfw_people.target
print(y.shape)

#print(X)
#print(y)





X = X.astype('float32') # Rescale it from 255 to 0-1.
X = X/255.

# one hot encode
enc_y = to_categorical(y)


#Splitting data into train and test data
train_X,test_X,train_y,test_y = train_test_split(X, enc_y, test_size=0.3)
#print(train_X.shape)


uniqueClasses=np.unique(y)
NuniqueClasses=len(uniqueClasses)

img_shape = train_X[0].shape
print('The image shape is: {}'.format(img_shape))

class_amount = NuniqueClasses
print('The amount of classes is: {}'.format(class_amount))

#%%
batch_size = 64
epochs = 100

#Get back the convolutional part of a VGG network trained on ImageNet
model_vgg16_conv = VGG16(weights='imagenet', include_top=False)
model_vgg16_conv.summary()

#Create your own input format (here 3x200x200)
input = Input(shape=img_shape,name = 'image_input')

#Use the generated model 
output_vgg16_conv = model_vgg16_conv(input)

#Add the fully-connected layers 
x = Flatten(name='flatten')(output_vgg16_conv)
x = Dense(4096, activation='relu', name='fc1')(x)
x = Dense(4096, activation='relu', name='fc2')(x)
x = Dense(class_amount, activation='softmax', name='predictions')(x)

#Create your own model 
model = Model(input=input, output=x)

#In the summary, weights and layers from VGG part will be hidden, but they will be fit during the training
model.summary()

learningrate = 1e-2
adagrad = keras.optimizers.Adagrad(lr=learningrate, epsilon=None, decay=0.0005)
#model.compile(loss='categorical_crossentropy', optimizer=sgd)
model.compile(loss='categorical_crossentropy',
              optimizer=adagrad,
              metrics=['accuracy'])

history = model.fit(train_X, train_y,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          shuffle=True,
          validation_split=0.2)

score = model.evaluate(test_X, test_y, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1]*100)













