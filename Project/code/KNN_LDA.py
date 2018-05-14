import h5py
import scipy as spy
import numpy as np
import math
import matplotlib.pyplot as plt
#import tensorflow as tf
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
#from more_itertools import locate

#functions


def indices(lst, element):
    result = []
    offset = -1
    while True:
        try:
            offset = lst.index(element, offset+1)
        except ValueError:
            return result
        result.append(offset)



def genRand(min, max, sampleSize):
    answer=[]
    answerSize=0
    while answerSize < sampleSize:
        r = np.random.randint(min, max)
        if r not in answer:
            answerSize += 1
            answer.append(r)
    return answer


#load the data 



f = h5py.File('/Users/Marike/Documents/MATLAB/iriscode/database_segmented.mat', 'r');
 
print("Keys: %s" % f.keys())
a_group_key = list(f.keys())[3]
featureVector = np.array(f[a_group_key])
featureVector = featureVector.transpose()
p=featureVector.shape
print(p)
#print(featureVector)
 
import csv
with open('/Users/Marike/Documents/MATLAB/iriscode/label.csv', 'r') as f:
   reader = csv.reader(f)
   label = [row for row in reader]


#set general parameters

k_fold_param=5
PPTrainSize=5 

#split data in train and validation sets

label_list = [item for sublist in label for item in sublist]#make a it flatlist
uniqueClasses=np.unique(label_list)
NuniqueClasses=len(uniqueClasses)

np.random.seed(42)

TrainData=np.empty((0,p[1]))
TrainIndices=[]
TrainLabel=[] 

for s in range(0,NuniqueClasses):
    globals()['ClassSamples%s' % s]=indices(label_list,uniqueClasses[s])
    globals()['Rand%s' % s]=genRand(0,len(eval('ClassSamples' + str(s))),PPTrainSize)

    for t in range(0, len(eval('Rand' + str(s)))):
        TrainIndices.append(eval('ClassSamples' + str(s))[eval('Rand' + str(s))[t]])
        TrainData=np.append(TrainData,[featureVector[eval('ClassSamples' + str(s))[eval('Rand' + str(s))[t]]]],axis=0)
        TrainLabel.append(label_list[eval('ClassSamples' + str(s))[eval('Rand' + str(s))[t]]])
ValiLabel=[]#Create a list for the labels of the validation samples
ValiData=np.empty((0,p[1]))#Create an array for the validation samples
AllI=np.arange(p[0])
ValidationIndices=np.setdiff1d(AllI,TrainIndices)#find all indices for the ones not in train set

for NValiSample in range(0,len(ValidationIndices)):
    ValiLabel.append(label_list[ValidationIndices[NValiSample]])
    ValiData=np.append(ValiData,[featureVector[ValidationIndices[NValiSample]]],axis=0)

print(AllI)
print(ValidationIndices)
print(TrainData.shape)
print(TrainIndices)
print(ValidationIndices)


#set parameters KNN

NNeighbours=1


#classification using KNN
neigh = KNeighborsClassifier(n_neighbors=NNeighbours) #initiate KNN classifier 
neigh.fit(TrainData,TrainLabel) #train classifier
PredictedClass_KNN=neigh.predict(ValiData) #classify using trained classifier
PredictionCorrectness_KNN=PredictedClass_KNN==ValiLabel #compare to known labels
KNN_accuracy=PredictionCorrectness_KNN.sum()/len(ValiLabel)#calculate accuracy
print('KNN accuracy:',KNN_accuracy)

#set parameters LDA

#classification using LDA
Linear=LDA(solver='svd')
Linear.fit(TrainData,TrainLabel)
PredictedClass_LDA=Linear.predict(ValiData)
PredictionCorrectness_LDA=PredictedClass_LDA==ValiLabel
LDA_accuracy=PredictionCorrectness_LDA.sum()/len(ValiLabel)#calculate accuracy
print('LDA accuracy:',LDA_accuracy)


