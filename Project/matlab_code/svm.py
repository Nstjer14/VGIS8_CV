# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 10:04:54 2018

@author: Shaggy
"""
import h5py
import numpy as np

f = h5py.File('database_segmented.mat', 'r');

print("Keys: %s" % f.keys())
a_group_key = list(f.keys())[3]
featureVector = np.array(f[a_group_key])
featureVector = featureVector.transpose()


import csv
with open('label.csv', 'r') as f:
   reader = csv.reader(f)
   label = [row for row in reader]

'''   
## Code to count number of uniques in label 
mylist = label
mylist2 = []
for thing in mylist:
    thing = tuple(thing)
    mylist2.append(thing)
set(mylist2)   
'''

from sklearn.model_selection import KFold # import KFold

kf = KFold(n_splits=5) # Define the split - into 5 folds
#kf.get_n_splits(featureVector) # returns the number of splitting iterations in the cross-validator 
'''
for train, test in kf.split(featureVector):
    train_data = np.array(featureVector)[train]
    test_data = np.array(featureVector)[test]
    #train_label = label[train]
    #test_label = label[test]
    print(train)
    print(test)
    
'''    
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(featureVector, label, train_size=35,stratify=label,random_state=42)

#from sklearn import svm
#from sklearn.metrics import accuracy_score
#clf = svm.SVC()
#clf.fit(X_train,y_train)
#predicted = clf.predict(X_test)
#print(accuracy_score(y_test, predicted))
'''
This part is about training and seeing how well the different classifiers
perform with k=5 K-fold cross validation. We use 20% of the data for training
'''
from sklearn import svm
from sklearn.model_selection import cross_val_score
clf = svm.SVC(kernel='linear')
scores = cross_val_score(clf, X_train, np.ravel(y_train), cv=kf)
print("Accuracy for linear: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

clf2 = svm.SVC(kernel='poly')
scores2 = cross_val_score(clf2,X_train, np.ravel(y_train), cv=kf)
print("Accuracy for quadratic: %0.2f (+/- %0.2f)" % (scores2.mean(), scores2.std() * 2))

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis()
scores3 = cross_val_score(lda, X_train, np.ravel(y_train), cv=kf)
print("Accuracy for LDA: %0.2f (+/- %0.2f)" % (scores3.mean(), scores3.std() * 2))

from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
myList = list(range(1,25))
neighbors = list(filter(lambda x: x % 2 != 0, myList))
cv_scores = []
for k in neighbors:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, np.ravel(y_train), cv=kf, scoring='accuracy')
    cv_scores.append(scores.mean())
    
MSE = [1 - x for x in cv_scores]
optimal_k = neighbors[MSE.index(min(MSE))]
print ("The optimal number of neighbors is %d" % optimal_k)
plt.plot(neighbors, MSE)
plt.xlabel('Number of Neighbors K')
plt.ylabel('Misclassification Error')
plt.show()

knn2 = KNeighborsClassifier(n_neighbors=optimal_k)
scores4 = cross_val_score(knn2, X_train, np.ravel(y_train), cv=kf, scoring='accuracy')
print("Accuracy for KNN k =",optimal_k,": ",scores4.mean(), "+/-", scores4.std() * 2)

'''
Here we train the model with the 20% data and validate it on the rest of the 80% 
'''
clf_model = svm.SVC(kernel='linear').fit(X_train,np.ravel(y_train))
print("Accuracy of validation on linear SVM: ",clf_model.score(X_test,np.ravel(y_test)))

clf2_model = svm.SVC(kernel='poly').fit(X_train,np.ravel(y_train))
print("Accuracy of validation on quadratic SVM: ",clf2_model.score(X_test,np.ravel(y_test)))

lda_model = LinearDiscriminantAnalysis().fit(X_train,np.ravel(y_train))
print("Accuracy of validation on LDA: ",lda_model.score(X_test,np.ravel(y_test)))

knn_model = KNeighborsClassifier(n_neighbors=optimal_k).fit(X_train,np.ravel(y_train))
print("Accuracy of validation on KNN k=1: ",knn_model.score(X_test,np.ravel(y_test)))

