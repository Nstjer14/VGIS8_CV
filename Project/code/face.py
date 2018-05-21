#LFW database handling

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

batch_size = 256
epochs = 100

input_shape = img_shape
num_classes = class_amount

print("Shape of train_X",train_X.shape)
print("Shape of train_y",train_y.shape)
print("Shape of test_X",test_X.shape)
print("Shape of test_y",test_y.shape)



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
model.add(Dropout(0.25))
model.add(Dense(num_classes, activation='softmax',))

learningrate = 1e-2
adagrad = keras.optimizers.Adagrad(lr=learningrate, epsilon=None, decay=0.0005)
#model.compile(loss='categorical_crossentropy', optimizer=sgd)
model.compile(loss='categorical_crossentropy',
              optimizer=adagrad,
              metrics=['accuracy'])
model.summary()
#print(model.get_config())


history = model.fit(train_X, train_y,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          shuffle=True,
          validation_split=0.2)

score = model.evaluate(test_X, test_y, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1]*100)



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



#%%


if not os.path.isdir(pic_save_dir):
    os.makedirs(pic_save_dir)
pic_path = os.path.abspath(pic_save_dir)
plt.savefig(pic_path + '/acc.pdf')
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
plt.savefig(pic_path + '/loss.pdf')
print('Saved graphs at %s ' % pic_path)

plt.show()