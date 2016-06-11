#!/usr/bin/python

import numpy as np
import sys
import pickle
from pandas import DataFrame, Series
import pandas as pd

sys.path.append("tools/")

my_data = pickle.load(open("my_dataset.pkl", "r"))
my_clf = pickle.load(open("my_classifier.pkl", "r"))
my_features = pickle.load(open("my_feature_list.pkl", "r"))

print my_data
print my_clf
print my_features