#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
#sys.path.append("../tools/")
sys.path.insert(1, '/Users/John/Documents/GitHub/ud120-projects/tools')
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###

from sklearn import svm

clf = svm.SVC(kernel = 'rbf', C = 10000.0)
clf.fit(features_train[:len(features_train)/100], labels_train[:len(labels_train)/100])
pred = clf.predict(features_test)

print(str(clf.score(features_test, labels_test)))
print(pred[10])
print(pred[26])
print(pred[50])
print(len(pred))
print(pred.sum())
#########################################################


