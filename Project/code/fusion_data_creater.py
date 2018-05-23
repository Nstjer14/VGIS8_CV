#import load_images_2_python
import pandas as pd
from collections import Counter
from sklearn.datasets import fetch_lfw_people
import matplotlib.pyplot as plt
import numpy as np

#%% Iris Data load
dataFrame_iris = pd.read_pickle("pythonDatabase")

counts = Counter(dataFrame_iris.label)
discardList = []
minNumOfImages = 10
for iris_name,value in counts.items():
    if value<=minNumOfImages:
        discardList.append(iris_name)
dataFrame_iris = dataFrame_iris[~dataFrame_iris['label'].isin(discardList)]
print("Classes' with less than %.f images discarded in total are %.f : " % (minNumOfImages,len(discardList)),discardList)

temp_for_reshape = dataFrame_iris.image.values
img_dim = temp_for_reshape[1].shape
iris_imageVector = []

for i in temp_for_reshape:
    iris_imageVector.append(np.array(i.reshape(img_dim)))
iris_imageVector = np.asarray(iris_imageVector)

iris_label = dataFrame_iris.label.tolist() # The list coming from dataFrame is already in the correct format.

#%% Face data load

lfw_people = fetch_lfw_people(min_faces_per_person=10, 
                              slice_ = (slice(61,189),slice(61,189)),
                              resize=0.5, color = True)

face_data = lfw_people.images
print("Original data shape: ", face_data.shape)
test_image = face_data[1]
plt.imshow(test_image.astype(np.uint8), interpolation='nearest')
plt.axis('off')


# access the class labels
face_label = lfw_people.target
print("Original label shape: ",face_label.shape)
dataFrame_face = pd.DataFrame({'image':list(face_data),'label':face_label})
#%%
unique_iris = set(iris_label)
unique_iris = list(unique_iris)
unique_face = set(face_label)
unique_face = list(unique_face)

col_names =  ['iris', 'face', 'label']
fusionDataframe = pd.DataFrame(columns=col_names)
fusionData = []
for i in range(0,len(unique_iris)):
    current_label =  unique_iris[i]
    iris_temp = dataFrame_iris[dataFrame_iris['label']==unique_iris[i]]
    iris_temp = iris_temp.reset_index()
    face_temp =dataFrame_face[dataFrame_face['label']==unique_face[i]]
    face_temp = face_temp.reset_index()
    iris_temp_length = len(iris_temp)
    face_temp_length = len(face_temp)
    number_of_samples = min(iris_temp_length,face_temp_length)
    #label = np.empty(number_of_samples)
    #label.fill(i)
    for j in range(0,number_of_samples):
        fusionData.append({'iris':iris_temp.image[j],'face':face_temp.image[j],'label':i})
    #fusionDataframe.append(iris_temp[0:number_of_samples],face_temp[0:number_of_samples],label)

fusionDataframe = pd.DataFrame(fusionData)
    
#temp = fusionDataframe.face[10]
#plt.imshow(temp.astype(np.uint8), interpolation='nearest')


