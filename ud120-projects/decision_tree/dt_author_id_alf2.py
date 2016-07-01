#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
from sklearn.metrics import accuracy_score

from sklearn import tree

# clf_minsplt40 = tree.DecisionTreeClassifier(min_samples_split = 40)
# clf_minsplt40.fit(features_train, labels_train)
# pred40 = clf_minsplt40.predict(features_test)
# acc_min_samples_split_40 = accuracy_score(pred40, labels_test)
# print acc_min_samples_split_40

print len(features_train[0])
#########################################################


