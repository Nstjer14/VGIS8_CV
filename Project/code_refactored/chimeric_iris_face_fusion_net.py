import general_cnn_functions as general_cnn
import iris_cnn as iris_cnn_methods
import VGG16_face_cnn as face_cnn_methods
import fusion_data_creater as chimerc_data_module
from keras.layers.merge import Concatenate
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Model
import keras

import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split




import numpy as np

def getChimericLabel():
    '''
    Gets the chimeric database base and returns it the chimeric label 
    
    output:
        chimeric_label = The chimeric labels extracted from the chimericdataframe in a list type
        
    '''
    dataFrame = chimerc_data_module.getChimericDatabase()
    
    chimeric_label = dataFrame['chimeric_label']
    chimeric_label = chimeric_label.tolist() # The list coming from dataFrame is already in the correct format.
    chimeric_label = np.asarray(chimeric_label)
    chimeric_label = chimeric_label.astype(int)
    return chimeric_label



def createChimericCnnArchitecture(face_data,iris_data,number_of_classes):
    '''
    This creates the model for CNN for face recognition. To remove the outer layer model.layers.pop() can be used 
    input:
        train_data = the training data. Used to get the shape of the data for the input layer
        number_of_classes = the number of classes for classification used for the last layer  layer
    
    output:
        model= the architecture of the VGG16 face CNN model.
    '''
    
    face_cnn = face_cnn_methods.createFaceCnnArchitecture(face_data,number_of_classes)
    iris_cnn = iris_cnn_methods.createIrisCnnArchitecture(iris_data,number_of_classes)
    
    face_cnn.layers.pop() # Removes outer categorisation layer
    iris_cnn.layers.pop() # Removes outer categorisation layer
    #face_cnn.summary()
    #iris_cnn.summary()
    face_out = face_cnn.layers[-1].output
    iris_out = iris_cnn.layers[-1].output
    
    merged_layer = Concatenate()([iris_out, face_out])
    fc1_merged = Dense((1024*2), activation='relu',name = 'merged_fc1')(merged_layer)
    drop1 = Dropout(0.5)(fc1_merged)
    fc2_merged = Dense((1024*2), activation='relu',name = 'merged_fc2')(drop1)
    drop2 = Dropout(0.5)(fc2_merged)
    
    output_merged = Dense(number_of_classes, activation='softmax')(drop2)
    
    input_vgg16 = face_cnn.layers[0].input
    iris_model_in = iris_cnn.layers[0].input
    merged_model = Model(inputs=[iris_model_in, input_vgg16], outputs=output_merged)
    return merged_model


def pandaObjectToNumpy(temp_for_reshape):
    img_dim = temp_for_reshape[1].shape
    imageVector = []
    
    for i in temp_for_reshape:
        imageVector.append(np.array(i.reshape(img_dim)))
    imageVector = np.asarray(imageVector)
    return imageVector
    

def shuffleChimericData(iris_images,face_images,chimeric_label):
    '''
    This combines the iris, face and labels into a dataframe to shuffle it and then splits it back up into
    their respective numpy arrays. This is to ensure a correct suffleing of the data. It also returns the merged dataframe
    input:
        iris_images = the iris images in a numpy array
        face_images = the face images in a numpy array
        chimeric_label = the chimeric labels in a list form
        
    output:
        iris_img = the shuffled iris images in a numpy array
        face_img = the shuffled face images in a numpy array
        label = the shuffled chimeric labels in a list
        df = all of above in a dataframe
    '''
    df = pd.DataFrame({'iris_image':list(iris_images),'face_image':list(face_images),'chimeric_label':chimeric_label})
    df = shuffle(df)
    label = df['chimeric_label']
    label = label.tolist()
    iris_img = df['iris_image'].values
    iris_img = pandaObjectToNumpy(iris_img)
    face_img = df['face_image'].values
    face_img = pandaObjectToNumpy(face_img)
    return iris_img,face_img,label, df

def loadChimericDataInArrays():
    '''
    Loads the chimeric pandas dataframe and splits it into iris, face and chimeric label
    
    output:
        iris_imageVector = the iris images in a numpy array
        lfw_people = the face images in a numpy array
        chimeric_label = the chimeric labels in a list
        
    '''
    chimeric_label = getChimericLabel()
    #chimeric_database = chimerc_data_module.getChimericDatabase()
    irisDataFrame, iris_label = iris_cnn_methods.chimericLoadDataAndLabels()
    iris_imageVector = iris_cnn_methods.resizeImagesToArray(irisDataFrame)
    lfw_people,face_label = face_cnn_methods.chimericLoadDataAndLabels()

    return iris_imageVector, lfw_people, chimeric_label
    

def prepareChimericData():
    '''
    This function loads the chimeric data, splits it into numpy arrays, creates the appropriate patches and merges
    the data back into a pandas dataframe.
    
    output:
        chimeric_dataframe = a dataframe of the iris, face and chimeric labels after patches have been generated
        chimeric_label = the chimeric labels in a list form
    '''
    iris_imageVector, lfw_people, chimeric_label = loadChimericDataInArrays()
    temp_label = chimeric_label
    iris_imageVector_patches, chimeric_label_patches = general_cnn.makePatches(iris_imageVector,chimeric_label)
    lfw_people_patches, temp_label_patches = face_cnn_methods.makePatches(lfw_people,temp_label)
    iris_img,face_img,chimeric_label, chimeric_dataframe = shuffleChimericData(iris_imageVector_patches,lfw_people_patches,chimeric_label_patches)
    return chimeric_dataframe, chimeric_label

def splitChimericData(Test_size = 0.2):
    '''
    Loads all the prepared data, onehot encodes the labels and performs a splits of the data from the dataframe.
    The data is split into training (0.6), training (0.2) and validation (0.2). The is then converted from pandas object to numpy array
    
    output:
        train_iris_X
        train_face_X
        test_iris_X
        test_face_X
        validation_iris_X
        validation_face_X
        train_label
        test_label
        valid_label
        NuniqueClasses = number of classes

    '''
    chimeric_dataframe, chimeric_label = prepareChimericData()
    NuniqueClasses = len(np.unique(chimeric_label))
    chimeric_label_label_onehot = general_cnn.onehotEncodingLabels(chimeric_label)
    chimeric_dataframe = chimeric_dataframe.drop(['chimeric_label'],axis=1)
    train_X,test_X,train_label,test_label = train_test_split(chimeric_dataframe, chimeric_label_label_onehot, test_size=Test_size, random_state=13)
    test_X,valid_X,test_label,valid_label = iris_cnn_methods.valFromTestSplit(test_X,test_label,Test_size = 0.5)
    
    train_iris_X =  pandaObjectToNumpy(train_X['iris_image'].values)
    train_face_X =  pandaObjectToNumpy(train_X['face_image'].values)
    
    test_iris_X =  pandaObjectToNumpy(test_X['iris_image'].values)
    test_face_X =  pandaObjectToNumpy(test_X['face_image'].values)
    
    validation_iris_X =  pandaObjectToNumpy(valid_X['iris_image'].values)
    validation_face_X =  pandaObjectToNumpy(valid_X['face_image'].values)
    
    return train_iris_X,train_face_X, test_iris_X, test_face_X, validation_iris_X, validation_face_X, train_label, test_label, valid_label, NuniqueClasses

        


def trainingFusionNet(fusion_net,train_iris_X,train_face_X,train_label,validation_iris_X,validation_face_X,validation_label,Batch_size = 32,Epoch = 50,Learningrate = 1e-3):
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
    model = fusion_net
  
    print("Shape of train_iris_X",train_iris_X.shape)
    print("Shape of train_face_X",train_face_X.shape)
    print("Shape of train_label",train_label.shape)

    print("Shape of validation_iris_X",validation_iris_X.shape)
    print("Shape of validation_face_X",validation_face_X.shape)
    print("Shape of valid_label",valid_label.shape)
    
    batch_size = Batch_size
    epochs = Epoch
    
    learningrate = Learningrate
    adagrad = keras.optimizers.Adagrad(lr=learningrate, epsilon=None, decay=0.0005)
    model.compile(loss='categorical_crossentropy',
                  optimizer=adagrad,
                  metrics=['accuracy'])
    model.summary()
    
    history = model.fit([train_iris_X, train_face_X],
                     train_label,
                     batch_size=batch_size,
                     epochs=epochs,
                     verbose=1,
                     shuffle=True,
                     validation_data = ([validation_iris_X, validation_face_X],validation_label))
    
    return model,history

if __name__ == '__main__':
    chimeric_dataframe, chimeric_label = prepareChimericData()
    train_iris_X,train_face_X, test_iris_X, test_face_X, validation_iris_X, validation_face_X, train_label, test_label, valid_label, number_of_classes = splitChimericData()


    chimeric_fusion_model = createChimericCnnArchitecture(train_face_X,train_iris_X,number_of_classes)
    chimeric_fusion_model,history = trainingFusionNet(chimeric_fusion_model,train_iris_X,train_face_X,train_label,validation_iris_X,validation_face_X,valid_label,Batch_size = 32,Epoch = 50,Learningrate = 1e-3)
    #score = general_cnn.evaluateModel(chimeric_fusion_model,[test_iris_X, test_face_X],test_label)
    #plt_acc,plt_val = general_cnn.plotHistory(history)
    #general_cnn.saveModel(chimeric_fusion_model,score,plt_acc,plt_val,Model_name='chimeric_fusion_cnn')     
    

    
    pass