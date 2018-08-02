import iris_cnn as iris_cnn_methods
import VGG16_face_cnn as face_cnn_methods
import pandas as pd
from matplotlib.pyplot import imshow, pause
from sklearn.utils import shuffle
import numpy as np

def loadDatabases():
    iris_data, iris_label = iris_cnn_methods.loadIrisDatabase()
    face_data, face_label = face_cnn_methods.load_lfw()
    return iris_data, iris_label, face_data, face_label
    

def lowestNumberOfImages(dataframe1,dataframe2):
    '''
    Returns the smallest length as the number of usable images
    '''
    len1 = dataframe1.shape[0]
    len2 = dataframe2.shape[0]
    if len1 >= len2:
        num_images = len2
    else:
        num_images = len1
    
    return num_images   

def listOfUnique(label):
    '''
    input:
        label = list of labels
    output:
        uniqueList = a list of number of unique labels in the input list
    
    '''
    uniqueSet = set(label)
    uniqueList = list(uniqueSet)
    return uniqueList

def getMostDataPossible():
    '''
    Compares the iris database and face databases number of classes. Then chooses the number of classes
    to maximize the number of images in each class. Renames columns of the output.
    
    output:
        maxIrisFrame = panda dataframe containing maximum possible iris images
        maxIrisLabel = list containing the labels of maximum possible iris images in descending order
        maxFaceFrame = panda dataframe containing maximum possible face images
        maxFaceLabel = list containing the labels of maximum possible face images in descending order
    '''
    iris_data, iris_label, face_data, face_label = loadDatabases()
    iris_data = iris_data.drop(['full_path','image_number','featureVector'],axis=1)
    face_data = pd.DataFrame({'image':list(face_data),'label':face_label})
    unique_iris = listOfUnique(iris_label)
    unique_face = listOfUnique(face_label)
    numberOfClasses = min(len(unique_iris),len(unique_face))
    
    iris_frequency = iris_data['label'].value_counts()
    face_frequency = face_data['label'].value_counts()
    
    maxNumberIris= iris_frequency.iloc[0:numberOfClasses]
    maxNumberFace = face_frequency.iloc[0:numberOfClasses]
    maxIrisLabel = maxNumberIris.index.tolist() 
    maxFaceLabel = maxNumberFace.index.tolist() 
    maxIrisFrame = iris_data[iris_data['label'].isin(maxIrisLabel)]
    maxFaceFrame = face_data[face_data['label'].isin(maxFaceLabel)]
    
    maxIrisFrame = maxIrisFrame.rename({'image':'iris_image','label':'iris_label'},axis=1)
    maxFaceFrame = maxFaceFrame.rename({'image':'face_image','label':'face_label'},axis=1)

    
    return maxIrisFrame,maxIrisLabel, maxFaceFrame, maxFaceLabel    

def cropDataFrame(maxIrisFrame,iris_label, maxFaceFrame, face_label):
    '''
    Takes a class from iris database and the face database and crops them so the the class that will be
    used for merging have the same amount of images in both. The amount of images is based on the smallest
    dataframe.
    
    
    input:
        maxIrisFrame = the dataframe containing all the iris images
        iris_label = the label for the class that is currently being cropped
        maxFaceFrame = the dataframe containg all the face images
        face_label = the label for the class that is currently being cropped
        
    output:
        current_irisFrame_cropped = the iris database after cropping
        current_faceFrame_cropped = the face database after cropping
        
    '''
    current_irisFrame = maxIrisFrame[maxIrisFrame['iris_label']==iris_label]
    current_faceFrame = maxFaceFrame[maxFaceFrame['face_label']==face_label]
    numberOfImages = lowestNumberOfImages(current_irisFrame,current_faceFrame)
    
    current_irisFrame_cropped = current_irisFrame.iloc[0:numberOfImages]
    current_faceFrame_cropped = current_faceFrame.iloc[0:numberOfImages]
        
    return current_irisFrame_cropped,current_faceFrame_cropped

def createNewChimericClass(irisDataFrame,faceDataFrame,chimeric_label):
    '''
    Takes an iris dataFrame and a face dataframe. They are horisontally concatanted and a new chimeric label is generated.
    '''
    irisDataFrame = irisDataFrame.reset_index(drop=True) # Trick to making it merge propperly. If index not reset we get NaNs
    faceDataFrame = faceDataFrame.reset_index(drop=True)
    chimeric_dataFrame = pd.concat([irisDataFrame,faceDataFrame],axis=1)
    chimeric_dataFrame['chimeric_label'] = chimeric_label
    return chimeric_dataFrame

def createChimericDatabase(shuffleData = True):
    '''
    Thie creates the chimeric database. Gets the maximum amount of data possible by taking the classes with most images. 
    Then crops each class so they have same amount images pairwise in each class. Then it merges classes into a 
    chimeric class with a new label and puts the chimeric classes in a dataframe.
    
    input: 
        shuffleData = by default true. If it is false the returned database will not be shuffled.
        
    output:
        chimeric_dataframe = the chimeric database of iris and face images
    '''
    maxIrisFrame,orderedMaxIrisLabel, maxFaceFrame, orderedMaxFaceLabel = getMostDataPossible()
    numberOfClasses = len(orderedMaxFaceLabel) # arbiratry which label we use, as they should be same length
    chimeric_dataframe = pd.DataFrame({'iris_image':[],'iris_label':[],'face_image':[],'face_label':[],'chimeric_label':[]})
    for i in range(0,numberOfClasses):
        current_irisFrame_cropped,current_faceFrame_cropped = cropDataFrame(maxIrisFrame,orderedMaxIrisLabel[i], maxFaceFrame, orderedMaxFaceLabel[i])
        chimeric_dataframe = pd.concat([chimeric_dataframe, createNewChimericClass(current_irisFrame_cropped,current_faceFrame_cropped,i)])
    if shuffleData == True:
        shuffle(chimeric_dataframe)
    
    return chimeric_dataframe

def getChimericDatabase():
    '''
    gets the chimeric database. Might seems unessecary, but it's more to keep good coding practive
    '''
    chimeric_dataframe = createChimericDatabase()
    return chimeric_dataframe
if __name__ == '__main__':
    #a = getChimericDataframe()
    chimeric_dataframe = getChimericDatabase()

    
    
    
    
    
    #iris_temp = iris_data[iris_data['label']==unique_iris[1]]
    #face_temp =face_data[face_data['label']==unique_face[2]]


    pass

