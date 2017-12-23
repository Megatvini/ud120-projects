#!/usr/bin/python

import sys
import pickle
import matplotlib.pyplot
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

all_features = ['salary', 'to_messages', 'deferral_payments', 'total_payments', 'exercised_stock_options',
                'bonus', 'restricted_stock', 'shared_receipt_with_poi', 'restricted_stock_deferred',
                'total_stock_value', 'expenses', 'loan_advances', 'from_messages', 'other',
                'from_this_person_to_poi', 'poi', 'director_fees', 'deferred_income',
                'long_term_incentive', 'email_address',
                'from_poi_to_this_person']

features_list = [
    'poi',
    'to_messages',
    'exercised_stock_options',
]

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


print('total data points', len(data_dict))
poi_count = sum([x['poi'] for x in data_dict.values()])
non_poi_count = sum([not x['poi'] for x in data_dict.values()])
print('poi:', poi_count, 'non-poi:', non_poi_count)

###  Task 2: Remove outliers
del data_dict['TOTAL']

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
def add_new_features(features):
    to_messages = features['to_messages']
    from_messages = features['from_messages']
    if to_messages != 'NaN' and from_messages != 'NaN':
        features['messages_ratio'] = float(to_messages) / float(from_messages)
    else:
        features['messages_ratio'] = 'NaN'
    return features


# my_dataset = {k: add_new_features(v) for k, v in data_dict.items()}
my_dataset = data_dict
print('my_dataset', len(my_dataset))

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys=True, remove_all_zeroes=False)
labels, features = targetFeatureSplit(data)
print('labels', len(labels))


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV

sss = StratifiedShuffleSplit(labels, 1000, random_state=0)
parameters = {'min_samples_leaf': [0.1, 0.5], 'splitter': ('best', 'random'), 'criterion': ('gini', 'entropy')}
clf = GridSearchCV(clf, parameters, cv=sss)
clf.fit(features, labels)
print(clf.best_params_)
clf = clf.estimator

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)