#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### your code goes here 
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.30, random_state=42)

cls = DecisionTreeClassifier()
cls.fit(X_train, y_train)
print cls.score(X_test, y_test)

print sum(y_test)
print len(y_test)
predictions = cls.predict(X_test)
print 'true positives', sum(map(lambda x: x[0] * x[1], zip(predictions, y_test)))
print 'precision', precision_score(y_test, predictions)
print 'recall', recall_score(y_test, predictions)
