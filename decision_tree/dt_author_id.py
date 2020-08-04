#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
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
from sklearn import tree
import numpy

clt = tree.DecisionTreeClassifier(min_samples_split=40)

clt = clt.fit(features_train, labels_train)

pred = clt.predict(features_test)

labels_test = numpy.array(labels_test)
labels_test = labels_test.reshape(-1,1)
pred = pred.reshape(-1,1)
acc = clt.score(pred, labels_test)

print(acc)

#########################################################


