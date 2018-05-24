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


#%%



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
print("Original data shape: ", X.shape)
test_image = X[1]
plt.imshow(test_image.astype(np.uint8), interpolation='nearest')
plt.axis('off')
#import matplotlib.image as mpimg
#image = mpimg.imread("chelsea-the-cat.png")
#plt.imshow(image)



# access the class labels
y = lfw_people.target
print("Original label shape: ",y.shape)

#print(X)
#print(y)

#%%
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
        batchImages.append(cv2.flip(j,0)) # adding horisontal flip
        batchLabels.append(y[i])
batchImages = np.asarray(batchImages)
batchLabels = np.asarray(batchLabels)

#%%
data = batchImages
label = batchLabels



# one hot encode
enc_y = to_categorical(label)


#Splitting data into train and test data
train_X,test_X,train_y,test_y = train_test_split(data, enc_y, test_size=0.3)
#print(train_X.shape)


uniqueClasses=np.unique(y)
NuniqueClasses=len(uniqueClasses)

img_shape = train_X[0].shape
print('The image shape is: {}'.format(img_shape))

class_amount = NuniqueClasses
print('The amount of classes is: {}'.format(class_amount))
if __name__ == '__main__':
    
    #%%
    batch_size = 64
    epochs = 50
    num_class = 10
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
    x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dropout(0.5)(x)
    x = Dense(class_amount, activation='softmax', name='predictions')(x)
    '''
    model = model_vgg16_conv
    model.layers.pop()
    model.outputs = [model.layers[-1].output]
    model.layers[-1].outbound_nodes = []
    model.add(Dense(num_class, activation='softmax'))
    '''
    #%%
    #Create your own model 
    model = Model(input=input, output=x)
    
    #In the summary, weights and layers from VGG part will be hidden, but they will be fit during the training
    model.summary()
    #for layer in model.layers[:10]:
    #    layer.trainable = False
        
    from keras.optimizers import SGD
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    
    #learningrate = 1e-3
    #adagrad = keras.optimizers.Adagrad(lr=learningrate, epsilon=None, decay=0.0005)
    #model.compile(loss='categorical_crossentropy',
    #              optimizer=adagrad,
    #              metrics=['accuracy'])
    
    history = model.fit(train_X, train_y,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              shuffle=True,
              validation_split=0.2)
    
    score = model.evaluate(test_X, test_y, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1]*100)
    
    
    
    #%%
    
    acc_round = str(round(score[1]*100,2))
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    namestamp = timestamp + 'acc_' + acc_round 
    model_name = namestamp + '_VGG16_face.h5'
    
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, model_name)
    model.save(model_path)
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
    
    
    
    
    
    
    
    
