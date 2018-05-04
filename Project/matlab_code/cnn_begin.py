#Basic CNN

import numpy as np
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense, Flatten
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt

import pandas as pd
import load_images_2_python

batch_size = 15
#num_classes = 62
epochs = 10
model_name = 'seq_CNN_trained_model.h5'


# Load the data from pythonDatabase that is generated in main_script. 
# Contains a pandas dataframe with the image, label and feature.
dataFrame = load_images_2_python.dataFrame
dataFrame = pd.read_pickle("pythonDatabase")

label = dataFrame.label
label_list = label.tolist()

#print(label_list)
uniqueClasses=np.unique(label_list)
num_classes=len(uniqueClasses)
print("Number of classes: ",num_classes)


#Splitting data into test, training and validation

for i in range (0, num_classes)
    



model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(512,64,3)),
    Flatten(),
    Dense(2, activation='softmax'),
    ])

#model.summary()