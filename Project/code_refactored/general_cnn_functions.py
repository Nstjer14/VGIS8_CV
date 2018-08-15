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
#import iris_face_merge_cnn_data_splitter as getData
from sklearn.utils import shuffle
import datetime
rd.seed(42)

#%%
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
import numpy as np
from sklearn.model_selection import train_test_split

from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

import cv2


#dataFrame = pd.read_pickle("pythonDatabase")
#dataFrame = shuffle(dataFrame)

def makePatches(dataframe,labels,PlotPatches=False):
    '''
    makes 10 patches from one image. 5 from patchGen.imagePatchGenerator5()
    and those are flipped so 5 are normal and 5 are horisontaly flipped.
    input:
        dataframe = a Pandas dataframe from loadIrisDatabase()
        labels = the labels from loadIrisDatabase()
    
    output:
        imageVector = all the patches that are generated in a numpy array
        label = the labels all the patches in a list type
        
    '''
    if (isinstance(dataframe,type(pd.DataFrame()))==True):
        dataFrame = dataframe
        label = labels
        resized_image = []
        for image in dataFrame.image:
            resized_image.append(cv2.resize(image, (64, 64)))
    else:
        dataFrame = dataframe
        label = labels
        resized_image = []
        for image in dataFrame:
            resized_image.append(cv2.resize(image[1], (64, 64)))
            
    
    batchImages = []
    batchLabels = []
    for i in range(0,len(resized_image)):
        batchesInTupples = patchGen.imagePatchGenerator5(resized_image[i],plotPatches=PlotPatches)
        for j in batchesInTupples:
            #plt.imshow(j, cmap='gray')
            batchImages.append(j)
            batchLabels.append(label[i])
            batchImages.append(cv2.flip(j,1)) # adding horisontal flip
            batchLabels.append(label[i])
            
    reshapeDims = batchImages[1].shape
    imageVector = np.asarray(batchImages)
    #plt.figure(figsize=[8,imageVector.shape[1]])
    #plt.imshow(imageVector[:,:,0], cmap='gray')

    imageVector = imageVector.reshape(imageVector.shape[0],1,58,58) # format it from (64,512) to (64,512,1) since it is an image with only 1 channel
    #imageVector = imageVector.reshape(imageVector.shape[0],1,64,512) # format it from (64,512) to (64,512,1) since it is an image with only 1 channel
    imageVector = imageVector.astype('float32') # Rescale it from 255 to 0-1.
    imageVector = imageVector/255.
    print('Training data shape after reshape : ', imageVector.shape)
    return imageVector, batchLabels

def onehotEncodingLabels(labels):
    '''
    This takes labels in a list form and returns it in a one hot encoded array
    input:
        labels = labels in a list form
        
    output:
        label_onehot = the labels in a one hot encoded fashion
    '''
    label = labels # set labels to be the batchLabels. This is a quick and dirty
    
    
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
    return label_onehot


def trainModelWithVal(cnn_model,x_Train,y_Train,Valid_X,Valid_Label,Batch_size = 128,Epoch = 50,Learningrate = 1e-2):
    '''
    Trains the model with validation data that is provided (not using validation_split)
    input:
        cnn_model = the cnn model
        x_Train = training data
        y_Train = training labels
        Valid_X = validation data
        Valid_Label = validation labels
        Batch_size = the batch size. it is by default set to 128
        Epoch = the training epichs. It is by default set to 50
        
    output:
        model = the trained model. Used for further training or testing
        history = the history of the trained model. Used for plotting
    '''
    model = cnn_model
    x_train = x_Train
    y_train = y_Train
    valid_X = Valid_X
    valid_label = Valid_Label
    
    if type(x_train) is np.ndarray:
        pass
    else:
        raise Exception('Training data is not a numpy array')
    
    if type(y_train) is np.ndarray:
        pass
    else:
        raise Exception("Training label is not a numpy array")    
    print("Shape of x_train",x_train.shape)
    print("Shape of y_train",y_train.shape)
    print("Shape of valid_X",valid_X.shape)
    print("Shape of valid_label",valid_label.shape)
    
    batch_size = Batch_size
    epochs = Epoch
    
    learningrate = Learningrate
    #adagrad = keras.optimizers.Adagrad(lr=learningrate, epsilon=None, decay=0.0005)
    #model.compile(loss='categorical_crossentropy', optimizer=sgd)
    sgd = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
    model.summary()
    
    history = model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              shuffle=True,
              verbose=1,
              validation_data=(valid_X,valid_label))
    return model,history

def trainModelValsplit(cnn_model,x_Train,y_Train,Batch_size = 128,Epoch = 50,Learningrate = 1e-2):
    '''
    Trains the model with validation data that is provided (not using validation_split)
    input:
        cnn_model = the cnn model
        x_Train = training data
        y_Train = training labels
        Valid_X = validation data
        Valid_Label = validation labels
        Batch_size = the batch size. it is by default set to 128
        Epoch = the training epichs. It is by default set to 50
        
    output:
        model = the trained model. Used for further training or testing
        history = the history of the trained model. Used for plotting
    '''    
    model = cnn_model
    x_train = x_Train
    y_train = y_Train

    
    if type(x_train) is np.ndarray:
        pass
    else:
        raise Exception('Training data is not a numpy array')
    
    if type(y_train) is np.ndarray:
        pass
    else:
        raise Exception("Training label is not a numpy array")    
    print("Shape of x_train",x_train.shape)
    print("Shape of y_train",y_train.shape)
    #print("Shape of valid_X",valid_X.shape)
    #print("Shape of valid_label",valid_label.shape)
    
    batch_size = Batch_size
    epochs = Epoch
    
    learningrate = Learningrate
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
              validation_split = 0.2)
    return model,history

def evaluateModel(cnn_model,X_test,Y_test):
    '''
    Evaluates the given model. Used with data the the model has not been trained on.
    Shows the loss and score.
    
    input:
        cnn_model = the model that is being evaluated
    
    output:
        score = a list containing the loss and the accuracy
    '''
    model = cnn_model
    x_test = X_test
    y_test = Y_test
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1]*100)
    return score

def plotAccuracy(History):
    '''
    Plots the accuracy of the training and validation of the model
    
    input:
        History = the history of the model.
    
    output:
        plt = the plt handle of the figure. Used for saving the plot
    '''
    fig1, ax_acc = plt.subplots()
    plt.plot(History.history['acc'])
    plt.plot(History.history['val_acc'])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Model - Accuracy')
    plt.legend(['Training', 'Validation'], loc='lower right')
    return plt

def plotLoss(History):
    '''
    Plots the loss of the training and validation of the model
    
    input:
        History = the history of the model.
    
    output:
        plt = the plt handle of the figure. Used for saving the plot  
    '''
    fig2, ax_loss = plt.subplots()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Model- Loss')
    plt.legend(['Training', 'Validation'], loc='upper right')
    plt.plot(History.history['loss'])
    plt.plot(History.history['val_loss'])
    plt.legend(['Training', 'Validation'], loc='lower right')
    return plt

def plotHistory(History):
    '''
    Plots the accuracy and validation of the training and validation of the model
    
    input:
        History = the history of the model.
    
    output:
        plt_acc = the plt handle of the accuracy figure. Used for saving the plot
        plt_val = the plt handle of the validation figure. Used for saving the plot
    '''    
    plt_acc = plotAccuracy(History)
    plt_val = plotLoss(History)
    return plt_acc,plt_val

def saveModel(model,score,acc_plt,val_plt,Model_name = 'unnamed'):
    '''
    Saves the model structure and weights so it can be reused. Also saves the 
    training and validation accuracy and loss plots. They ares saved in folders
    named "saved_models" and "saved_graphs" respectivaly. The naming convention
    used is YYYYMMDD-HHMMSS_acc_XX_model_name.h5
    
    input:
       cnn_model = the model that is saved
       Score = the list of accuracy and validation scores from evaluateModel()
       acc_plt = the training and validation accuracy plot
       val_plt = the training and validation loss plot
    '''
    
    save_dir = os.path.join(os.getcwd(), 'saved_models')
    pic_save_dir = os.path.join(os.getcwd(), 'saved_graphs')
    acc_round = str(round(score[1]*100,2))
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    namestamp = timestamp + 'acc_' + acc_round 
    model_name = namestamp + '_'+Model_name+'.h5'
    
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    
    if not os.path.isdir(pic_save_dir):
        os.makedirs(pic_save_dir)
    
    model_path = os.path.join(save_dir, model_name)
    model.save(model_path)
    print('Saved trained model at %s ' % model_path)
    
    
    pic_path = os.path.abspath(pic_save_dir)
    acc_plt.savefig(pic_path + '/'+model_name+'acc.pdf')
    print('Saved accuracy plot at %s ' % pic_path)

    pic_path = os.path.abspath(pic_save_dir)
    print(pic_path)
    val_plt.savefig(pic_path + '/'+model_name+'loss.pdf')
    print('Saved loss plot at %s ' % pic_path)
    
if __name__ == '__main__':
    print("There is nothing to gain by running this script on it's own")
    
    
    #exploreData(imageVector,label)

    pass