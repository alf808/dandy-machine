from sklearn.grid_search import GridSearchCV
from sklearn import cross_validation
print "Tune Classifiers\n"

## Tune decision tree via gridsearch

# Set up cross validator (will be used for tuning all classifiers)
cv = cross_validation.StratifiedShuffleSplit(labels,
                                            n_iter = 10,
                                             random_state = 42)
# set up estimator and pipeline, using PCA for feature selection
estimators = [('reduce_dim', PCA()),('dec_tree',dt_clf)]
dtclf = Pipeline(estimators)

# set up paramaters dictionary
dt_params = dict(reduce_dim__n_components=[perc_var],
              dec_tree__criterion=("gini","entropy"),
                  dec_tree__min_samples_split=[1,2,4,8,16,32],
                   dec_tree__min_samples_leaf=[1,2,4,8,16,32],
                   dec_tree__max_depth=[None,1,2,4,8,16,32])

# set up gridsearch
dt_grid_search = GridSearchCV(dtclf, param_grid = dt_params,
                          scoring = 'f1', cv =cv)

# pass data into into the gridsearch via fit
dt_grid_search.fit(features, labels)

print 'Decision tree tuning\n Steps: {0}\n, Best Parameters: {1}\n '.format(dtclf.steps,dt_grid_search.best_params_,dt_grid_search.best_score_)
# print sep2
# pick a winner
best_dtclf = dt_grid_search.best_estimator_

# best_dt_params = DecisionTreeClassifier(compute_importances=None, criterion='gini',
#                                         max_depth=2, min_samples_leaf=1, min_samples_split=8,
#                                         splitter='best')


## count the NaNs and print percentage of NaNs for each feature

my_df = pd.DataFrame(my_dataset).transpose() # turn columns to rows
# my_df = my_df.drop('email_address')
nan_counts_dict = {}
for column in my_df.columns:
    my_df[column] = my_df[column].replace('NaN',np.nan)
    nan_counts = my_df[column].isnull().sum()
    nan_counts_dict[column] = round(float(nan_counts)/float(len(my_df[column])) * 100,1)
df = pd.DataFrame(nan_counts_dict,index = ['percent_of_NaN']).transpose()
df = df.drop('email_address')
df.reset_index(level=0,inplace=True)
df = df.rename(columns = {'index':'feature'}).sort_values('percent_of_NaN', ascending=False)
df


split, train, transform, predict, validate/evaluate