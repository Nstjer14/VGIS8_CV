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
import datetime
import cv2
import imagePatchGenerator5_module as patchGen
import general_cnn_functions as general_cnn
import iris_cnn as cnn_functions
#import iris_face_merge_cnn_data_splitter as getData
#%%
rd.seed(42)

def load_lfw():
    # load images for individuals w/ 10+ images and produce centered 64x64 images from orig. 250x250 images
    lfw_people = fetch_lfw_people(min_faces_per_person=10, 
                                  slice_ = (slice(61,189),slice(61,189)),
                                  resize=0.5, color = True)
    images = lfw_people.images

    label = lfw_people.target
    
    
    return images, label
    
def check_lfw_images(lfw_people):
    #for name in lfw_people.target_names:
#    print(name)
    
    print("Original data shape: ", lfw_people.shape)
    test_image = lfw_people[1]
    plt.imshow(test_image.astype(np.uint8), interpolation='nearest')
    plt.axis('off')
    
   # print("Original label shape: ",label.shape)
    #import matplotlib.image as mpimg
    #image = mpimg.imread("chelsea-the-cat.png")
    #plt.imshow(image)

def makePatches(data,labels,PlotPatches=False):
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
    X = data
    y = labels
    
    X = X.astype('float32') # Rescale it from 255 to 0-1.
    X = X/255.

    batchImages = []
    batchLabels = []
    for i in range(0,len(X)):
        batchesInTupples = patchGen.imagePatchGenerator5(X[i])
        for j in batchesInTupples:
            #plt.imshow(j, cmap='gray')
            batchImages.append(j)
            batchLabels.append(y[i])
            batchImages.append(cv2.flip(j,1)) # adding horisontal flip
            batchLabels.append(y[i])
    batchImages = np.asarray(batchImages)
    batchLabels = np.asarray(batchLabels)
    print('Training data shape after reshape : ', batchImages.shape)
    return batchImages, batchLabels
    


def categoricalOnehotEncodingLabels(labels):
    '''
    This takes labels in a list form and returns it in a one hot encoded array
    input:
        labels = labels in a list form
        
    output:
        enc_y = the labels in a one hot encoded fashion
    '''
    enc_y = to_categorical(labels)
    return enc_y

def createFaceCnnArchitecture(train_data,number_of_classes):
    '''
    This creates the model for CNN for face recognition. To remove the outer layer model.layers.pop() can be used 
    input:
        train_data = the training data. Used to get the shape of the data for the input layer
        number_of_classes = the number of classes for classification used for the last layer  layer
    
    output:
        model= the architecture of the VGG16 face CNN model.
    '''
    #batch_size = 64
    #epochs = 50
    img_shape = train_data[0].shape
    num_class = number_of_classes

    #Get back the convolutional part of a VGG network trained on ImageNet
    model_vgg16_conv = VGG16(weights='imagenet', include_top=False)
    model_vgg16_conv.summary()
    
    #Create your own input format (here 3x200x200)
    input = Input(shape=img_shape,name = 'image_input')
    
    #Use the generated model 
    output_vgg16_conv = model_vgg16_conv(input)
    
    #Add the fully-connected layers 
    vgg_flat = Flatten(name='flatten')(output_vgg16_conv)
    face_dense_1 = Dense(4096, activation='relu', name='fc1')(vgg_flat)
    face_dense_1_drop = Dropout(0.5)(face_dense_1)
    face_dense_2 = Dense(4096, activation='relu', name='fc2')(face_dense_1_drop)
    face_dense_2_drop = Dropout(0.5)(face_dense_2)
    output_layer = Dense(num_class, activation='softmax', name='predictions')(face_dense_2_drop)
    model = Model(input=input, output=output_layer)
    '''
    model = model_vgg16_conv
    model.layers.pop()
    model.outputs = [model.layers[-1].output]
    model.layers[-1].outbound_nodes = []
    model.add(Dense(num_class, activation='softmax'))
    '''
    
    #In the summary, weights and layers from VGG part will be hidden, but they will be fit during the training
    model.summary()
    #for layer in model.layers[:10]:
    #    layer.trainable = False
        
    #from keras.optimizers import SGD
    #sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    #model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    return model





def splitDataFromlfw():
    '''
    This gets the data from the labeled faces in the wild (lfw), performs the nesecary operations and splits it.
    It returns train and validation data
    '''
    lfw_people,label =load_lfw()
    check_lfw_images(lfw_people)
    lfw_people_patches, label = makePatches(lfw_people,label)
    check_lfw_images(lfw_people_patches)
    label_onehot = categoricalOnehotEncodingLabels(label)
    NuniqueClasses = len(np.unique(label))
    print('Number of classes:', NuniqueClasses)

    train_X,valid_X,train_label,valid_label = train_test_split(lfw_people_patches, label_onehot, test_size=0.4,shuffle=True, random_state=13)
    return train_X,valid_X,train_label,valid_label,NuniqueClasses

def ValSplitIrisAcc():

    train_X,test_X,train_label,test_label,NuniqueClasses = splitDataFromlfw()
    test_X,valid_X,test_label,valid_label = cnn_functions.valFromTestSplit(test_X,test_label,Test_size = 0.5)
    model = createFaceCnnArchitecture(train_X,NuniqueClasses)
    model,history = general_cnn.trainModelWithVal(model,train_X,train_label,valid_X,valid_label,Batch_size = 64,Epoch = 50,Learningrate = 1e-2)
    score = general_cnn.evaluateModel(model,test_X,test_label)
    plt_acc,plt_val = general_cnn.plotHistory(history)
    general_cnn.saveModel(model,score,plt_acc,plt_val,Model_name='face_cnn')    

def trainWithoutVal():
    '''
    These settings with achieve 96,8% accuracy for face. It is done with using the automatic validation split, and not on the real validation data
    '''
    train_X,test_X,train_label,test_label,NuniqueClasses = splitDataFromlfw()
    model = createFaceCnnArchitecture(train_X,NuniqueClasses)
    model,history = general_cnn.trainModelValsplit(model,train_X,train_label,Batch_size = 32,Epoch = 50,Learningrate = 1e-3)
    score = general_cnn.evaluateModel(model,test_X,test_label)
    plt_acc,plt_val = general_cnn.plotHistory(history)
    general_cnn.saveModel(model,score,plt_acc,plt_val,Model_name='face_cnn')        
if __name__ == '__main__':
    #ValSplitIrisAcc()
    trainWithoutVal()
    pass