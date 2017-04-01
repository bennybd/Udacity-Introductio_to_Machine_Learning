#!/usr/bin/python

import sys
import pickle
import numpy as np
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
print
print "Task 1"
print "------"
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary','to_messages','deferral_payments','total_payments','exercised_stock_options','bonus','restricted_stock','shared_receipt_with_poi','restricted_stock_deferred','total_stock_value','expenses','loan_advances','from_messages','other','from_this_person_to_poi','director_fees','deferred_income','long_term_incentive','from_poi_to_this_person'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)

import csv
#with open('enron_data.csv', 'wb') as csvfile:
#    data_csv = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
#    row = ["name"]
#    for potential_feature in data_dict['ALLEN PHILLIP K']:
#        row += ['\'' + potential_feature + '\'']
#    data_csv.writerow(row)
#    for name in data_dict:
#        row = [value for key, value in data_dict[name].items()]
#        data_csv.writerow([name] + row)
from sklearn import preprocessing
#features_scaled = preprocessing.MinMaxScaler().fit_transform(features)
features_scaled = preprocessing.scale(features)

with open('features_data.csv', 'wb') as csvfile:
    data_csv = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
    row = []
    for feature in features_list:
        row += [feature]
            
    data_csv.writerow(row)
    for i in range(1,len(labels)):
        row = [labels[int(i)]]
        for feature in features_list[1:]:
            row += [features_scaled[int(i)][features_list.index(feature) - 1]]
        data_csv.writerow(row)


### Task 2: Remove outliers
print
print "Task 2"
print "------"
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
print
print "Task 3"
print "------"
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html
print
print "Task 4"
print "------"

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
print
print "Task 5"
print "------"

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.
print
print "Task 6"
print "------"

dump_classifier_and_data(clf, my_dataset, features_list)