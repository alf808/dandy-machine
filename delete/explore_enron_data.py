#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

def enron_data_set():
    import numpy as np
    import pickle
    from pandas import DataFrame, Series
    import pandas as pd
    enron_data = pickle.load(open("final_project_dataset.pkl", "r"))
    df_poi_names = pd.read_table("poi_names.txt")


    # feature_keys = enron_data['SKILLING JEFFREY K'].keys()

    
    return enron_data
    # return feature_keys

    # return npeople, nfeatures, npoi, total_poi, prentice, wc_to_poi, skilling_options, nemailadd, nsalary, n_missing_totalpayments, n_nan_poi_totalpayments

# print enron_data_set()

