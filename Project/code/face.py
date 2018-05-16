#LFW database handling

from __future__ import print_function
import tensorflow as tf
import numpy as np
import os
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
from datasets import dataset_utils
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split


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

#print(X)
#print(y)

#Splitting data into train and test data
train_X,test_X,train_y,test_y = train_test_split(X, y, test_size=0.3)
#print(train_X.shape)


uniqueClasses=np.unique(y)
NuniqueClasses=len(uniqueClasses)

img_shape = train_X[0].shape
print('The image shape is: {}'.format(img_shape))

class_amount = NuniqueClasses
print('The amount of classes is: {}'.format(class_amount))

batch_size = 128
epochs = 2

input_shape = img_shape
num_classes = class_amount


model = Sequential()
model.add(Conv2D(6, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape,
                 padding = "same"))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))
model.add(Conv2D(32,
                 kernel_size=(5, 5),
                 activation='relu',
                 padding = "same"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64,
                 kernel_size=(5, 5),
                 activation='relu',
                 padding = "same",))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256,
                 kernel_size=(5, 5),
                 activation='relu',
                 padding = "same",))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax',))

learningrate = 1e-2
adagrad = keras.optimizers.Adagrad(lr=learningrate, epsilon=None, decay=0.0005)
#model.compile(loss='categorical_crossentropy', optimizer=sgd)
model.compile(loss='categorical_crossentropy',
              optimizer=adagrad,
              metrics=['accuracy'])
model.summary()
print(model.get_config())


history = model.fit(train_X, train_y,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_split=0.1)

score = model.evaluate(test_X, test_y, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1]*100)







"""
face_dir = os.path.join(os.getcwd(), 'lfw_df/')
print(face_dir)

people_number = []


# Count number of photos of each individual. People number is a list of tuples 
# with each tuple composed of the name and the number of photos of a person.
for person in people_number:
    folder_path = face_dir + person
    num_images = len(os.listdir(folder_path))
    people_number.append((person, num_images))

# Sort the list of tuples by the number of images
people_number = sorted(people_number, key=lambda x: x[1], reverse=True)

# List Comprehension to determine number of people with one image
people_with_one_photo = [(person) for person, num_images in people_number if num_images==1]
print("Individuals with one photo: {}".format(len(people_with_one_photo)))
"""
"""
# Full deep-funneled images dataset
FACES_URL = "http://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz"
IMAGES_DOWNLOAD_DIRECTORY = os.getcwd()
print(IMAGES_DOWNLOAD_DIRECTORY)
IMAGES_DIRECTORY = "images/faces"
if not os.path.exists(IMAGES_DOWNLOAD_DIRECTORY):
    os.makedirs(IMAGES_DOWNLOAD_DIRECTORY)
# If the file has not already been downloaded, retrieve and extract it
if not os.path.exists(IMAGES_DOWNLOAD_DIRECTORY + "/lfw-deepfunneled.tgz"):
    dataset_utils.download_and_uncompress_tarball(FACES_URL, IMAGES_DOWNLOAD_DIRECTORY)

#folder_path = IMAGES_DOWNLOAD_DIRECTORY + '/lfw-deepfunneled/'
#print(folder_path)
people_number = []
# Count number of photos of each individual. People number is a list of tuples 
# with each tuple composed of the name and the number of photos of a person.
for person in people:
    print('enter')
    folder_path = IMAGES_DOWNLOAD_DIRECTORY + '/lfw-deepfunneled/' + person
    print(folder_path)
    num_images = len(os.listdir(folder_path))
    people_number.append((person, num_images))
 


# Sort the list of tuples by the number of images
people_number = sorted(people_number, key=lambda x: x[1], reverse=True)
# List Comprehension to determine number of people with one image
people_with_one_photo = [(person) for person, num_images in people_number if num_images==1]
print("Individuals with one photo: {}".format(len(people_with_one_photo)))
"""