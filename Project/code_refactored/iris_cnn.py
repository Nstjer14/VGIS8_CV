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
import general_cnn_functions as general_cnn
#import iris_face_merge_cnn_data_splitter as getData
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
'''
This module makes an iris cnn and trains is using pythonDatabase as a database.
functionality:
    loadIrisDatabase() = Loads the database and discards
    resizeImagesToArray(database) = resizes images to the wanted size
    exploreData(data,label_in) = used to plot the first data points
    makePatches(dataframe,labels,PlotPatches=False) = makes 5 image patches from 1 image
    onehotEncodingLabels(labels) = performs one hot encoding
    splitDataFromDatabase() = splits the data into training, validation and testing.
    
'''

#dataFrame = pd.read_pickle("pythonDatabase")
#dataFrame = shuffle(dataFrame)

def checkIfpandas(data):
    if (isinstance(data,type(pd.DataFrame()))==True):
        pass
    else:
        raise Exception('dataFrame is not a pandas dataframe')    

def loadIrisDatabase():
    '''
    Loads the database pythonDatabase from the folder that the script is in that contains the normalised iris iamges.
    Classes with less than 10 images are discarded.
    
    output:
        dataFrame = The database with dropped images as a Pandas dataframe
        label = The labels extracted from dataFrame in a list type
    
    '''

    # Comment this part in to use the data before the merger
    # Dropping classes with less than 10 images
    try:
        dataFrame = pd.read_pickle("pythonDatabase") 
    except Exception as e:
        #raise SystemExit(0)
        raise Exception('Could not locate pythonDatabase. Check folder if it is there')
    
    checkIfpandas(dataFrame)
    
    dataFrame = shuffle(dataFrame)
    counts = Counter(dataFrame.label)
    discardList = []
    minNumOfImages = 10
    for iris_name,value in counts.items():
        if value<=minNumOfImages:
            discardList.append(iris_name)
    dataFrame = dataFrame[~dataFrame['label'].isin(discardList)]
    print("Classes' with less than %.f images discarded in total are %.f : " % (minNumOfImages,len(discardList)),discardList)
    label = dataFrame.label
    label = label.tolist() # The list coming from dataFrame is already in the correct format.
    return dataFrame, label

def resizeImagesToArray(database):
    '''
    Resizes the images from a Pandas array to a numpy array.
    
    output:
        imageVector = the images from the Pandas dataframe as a numpy array.
    '''

    dataFrame = database    
    checkIfpandas(dataFrame)
    temp_for_reshape = dataFrame.image.values
    img_dim = temp_for_reshape[1].shape
    imageVector = []
    
    for i in temp_for_reshape:
        imageVector.append(np.array(i.reshape(img_dim)))
    imageVector = np.asarray(imageVector)
    return imageVector


def exploreData(data,label_in):
    '''
    This plots the data and prints the dimensions of the dataset.
    input:
        data = the images as a numpy array.
        label_in = the labels in a list type
    '''
    imageVector = data
    label = label_in
    if type(imageVector) is np.ndarray:
        pass
    else:
        raise Exception('imageVector is not a numpy array')
    
    if type(label) is list:
        pass
    else:
        raise Exception("label is not a list")
        

    print('Training data shape : ', imageVector.shape)
    uniqueClasses=np.unique(label)
    NuniqueClasses=len(uniqueClasses)
    print("Number of classes: ",NuniqueClasses)
    
    plt.figure(figsize=[8,imageVector.shape[1]])
    
    plt.subplot(121)
    plt.imshow(imageVector[0,:,:], cmap='gray')
    plt.title("Ground Truth : {}".format(label[0]))



def splitDataFromDatabase():
    '''
    This gets the data from the pythonDatabase, performs the nesecary operations and splits it.
    It returns train and validation data
    '''
    dataFrame, label = loadIrisDatabase()
    imageVector = resizeImagesToArray(dataFrame)
    exploreData(imageVector,label)
    imageVector, label = general_cnn.makePatches(dataFrame,label)
    label_onehot = general_cnn.onehotEncodingLabels(label)
    
    NuniqueClasses = len(np.unique(label))
    print('Number of classes:', NuniqueClasses)

    train_X,valid_X,train_label,valid_label = train_test_split(imageVector, label_onehot, test_size=0.2, random_state=13)
    return train_X,valid_X,train_label,valid_label,NuniqueClasses


def createIrisCnnArchitecture(train_data,number_of_classes):
    '''
    This creates the model for CNN for iris recognition. To remove the outer layer model.layers.pop() can be used 
    input:
        train_data = the training data. Used to get the shape of the data for the input layer
        number_of_classes = the number of classes for classification used for the last layer  layer
    
    output:
        model= the architecture of the iris CNN model.
    '''
    input_shape = train_data[0].shape
    num_classes = number_of_classes
    
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
    return model


    
if __name__ == '__main__':

    train_X,valid_X,train_label,valid_label,NuniqueClasses = splitDataFromDatabase()
    model = createIrisCnnArchitecture(train_X,NuniqueClasses)
    model,history = general_cnn.trainModelValsplit(model,train_X,train_label)
    score = general_cnn.evaluateModel(model,valid_X,valid_label)
    plt_acc,plt_val = general_cnn.plotHistory(history)
    general_cnn.saveModel(model,score,plt_acc,plt_val,Model_name='iris_cnn')
    
    
    #exploreData(imageVector,label)

    pass