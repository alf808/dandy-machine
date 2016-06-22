#!/usr/bin/python

import numpy as np
import sys
import pickle
from pandas import DataFrame, Series
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_selection import SelectKBest, SelectPercentile

sys.path.append("tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data
from auxiliary import computeFraction

# enron_data = pickle.load(open("final_project_dataset.pkl", "r"))
### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

# print data_dict
### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
# features_list = ['poi','salary'] # You will need to use more features
features_list = data_dict['SKILLING JEFFREY K'].keys()
features_list.remove('email_address')
features_list.remove('poi')
features_list = ['poi'] + features_list
# test_features = ['salary', 'bonus']

print features_list
# print data_dict
### Task 2: Remove outliers
## discussed in mini-project "outliers"

outliers = ['TOTAL', 'THE TRAVEL AGENCY IN THE PARK']

for point in outliers:
	data_dict.pop(point)

# print data_dict


### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list)
# data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Continue Feature Selection and dimensionality reduction via get_k_best

# assemble feature list
# my_features_list = ['poi'] + list(k_best_features.feature.values)

## get k (k represents number of features) best features
# k = 10
# k_best_features = enron_tools.get_k_best(data_dict,features_list,k)

# k_best = SelectKBest(k = k)
# k_best.fit(features, labels)
# scores = k_best.scores_
# pairs = zip(features_list[1:], scores)

#combined scores and features into a pandas dataframe then sort 
# k_best_features = pd.DataFrame(pairs, columns = ['feature','score'])
# k_best_features = k_best_features.sort('score', ascending = False)
    
    
#merge with null counts    
# df_nan_counts = get_nan_counts(dictionary)
# k_best_features = pd.merge(k_best_features, df_nan_counts, on = 'feature')  
    
#eliminate infinite values
# k_best_features = k_best_features[np.isinf(k_best_features.score) == False]
# print 'Feature Selection by k_best_features\n'
# print "{0} best features in descending order: {1}\n".format(k, k_best_features.feature.values[:k])
# print '{0}\n'.format(k_best_features[:k])
    
    
# return k_best_features[:k]

## scale extracted features
# scaler = preprocessing.MinMaxScaler()
# features = scaler.fit_transform(features)


# Set up cross validator (will be used for tuning all classifiers)
# cv = cross_validation.StratifiedShuffleSplit(tru,
#                                             n_iter = 10,
#                                              random_state = 42)

## Evaluate Final Adaboost Classifier

# load tuned classifier pipeline


# best_a_pipe = pickle.load(open('best_clf_pipe.pkl', "r") )


# print 'best_a_clf\n'
# best_a_pipe
# test_classifier(best_a_pipe,my_dataset,my_features_list)
# print data
# print labels, features
### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
gnb_clf = GaussianNB()

### Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
dt_clf = DecisionTreeClassifier()

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
# from sklearn.cross_validation import train_test_split
# features_train, features_test, labels_train, labels_test = \
#     train_test_split(features, labels, test_size=0.3, random_state=42)
import sklearn.cross_validation 
score = sklearn.cross_validation.cross_val_score( gnb_clf, features, labels )

print (score)

import sklearn.tree
import sklearn.ensemble
dtc_clf = sklearn.tree.DecisionTreeClassifier()
score = sklearn.cross_validation.cross_val_score( dtc_clf, features, labels )
print( score )
rfc_clf = sklearn.ensemble.RandomForestClassifier()
score = sklearn.cross_validation.cross_val_score( rfc_clf, features, labels )
print( score )
# test_classifier(gnb_clf, my_dataset, features_list, folds = 1000)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

# dump_classifier_and_data(clf, my_dataset, features_list)
