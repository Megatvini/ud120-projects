#!/usr/bin/python
# coding=utf-8

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

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

# How many data points (people) are in the dataset?
print len(enron_data)

# How many data points (people) are in the dataset?
print len(enron_data.values()[0])

# count the number of entries in the dictionary where data[person_name]["poi"]==1
print len([x for x in enron_data.values() if x['poi'] == 1])

# What is the total value of the stock belonging to James Prentice?
print enron_data['PRENTICE JAMES']['total_stock_value']

# How many email messages do we have from Wesley Colwell to persons of interest?
print enron_data['COLWELL WESLEY']['from_this_person_to_poi']

# What’s the value of stock options exercised by Jeffrey K Skilling?
print enron_data['SKILLING JEFFREY K']['exercised_stock_options']

# Of these three individuals (Lay, Skilling and Fastow), who took
# home the most money (largest value of “total_payments” feature)?
print max(
    enron_data['LAY KENNETH L']['total_payments'],
    enron_data['SKILLING JEFFREY K']['total_payments'],
    enron_data['FASTOW ANDREW S']['total_payments']
)

# How many folks in this dataset have a quantified salary?
# What about a known email address?
print len([x for x in enron_data.values() if x['salary'] != 'NaN'])
print len([x for x in enron_data.values() if x['email_address'] != 'NaN'])
