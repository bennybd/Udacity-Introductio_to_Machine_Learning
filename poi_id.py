#!/usr/bin/python

import sys
import pickle
import numpy as np
import math
import matplotlib.pyplot as plt

sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from mytools import scale

### Task 1: Select what features you'll use.
print
print "Task 1"
print "------"
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
'''
    restricted_stock_deferred       Only 2 point of data at non-poi
    loan_advances                   No data at all !!
    deferral_payments               Very little un-interesting data
    director_fees                   Very little un-interesting data
'''
features_list = ['poi','salary','to_messages','total_payments','exercised_stock_options','bonus','restricted_stock','shared_receipt_with_poi','total_stock_value','expenses','from_messages','other','from_this_person_to_poi','deferred_income','long_term_incentive','from_poi_to_this_person']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
data_dict.pop("TOTAL", 0)
data_dict.pop("LAY KENNETH L", 0)

#data = featureFormat(data_dict, features_list)
#labels, features = targetFeatureSplit(data)
data = featureFormat(data_dict, features_list, remove_NaN=False)
names = data[1]
labels, features = targetFeatureSplit(data)
data_range = range(len(labels))

#from sklearn import preprocessing
##features_scaled = preprocessing.MinMaxScaler().fit_transform(features)
#features_scaled = preprocessing.scale(features)
features_scaled = scale(features)

labels_sorted = np.array(labels)
features_sorted = features_scaled
indexes_sorted = np.array(data_range)

for feature in features_list[:0:-1]:   
    feature_index = features_list[1:].index(feature)
    feature_data = features_sorted[:,feature_index]
    for person_index, feature_value in enumerate(feature_data):
        if math.isnan(feature_value):
            features_sorted[person_index][feature_index] = 0

labels_inds = labels_sorted.argsort()[::-1]
labels_sorted = labels_sorted[labels_inds]
features_sorted = features_sorted[labels_inds]
indexes_sorted = indexes_sorted[labels_inds]

c0 =  1 / math.sqrt(2)
c = c0
print '{0:28s} {1:10s}  {2:10s}  {3:6s}   {4:30s}'.format('feature', 'value', 'unscaled', 'index', 'name')
print '{0:28s} {1:10s}  {2:10s}  {3:6s}   {4:30s}'.format('---------', '-------', '---------', '------', '-------')
for feature in features_list[1:]:
    clr = [c,(c + c0) % 1,(c + 2*c0) % 1]
    c = (c + 5*c0) % 1
    feature_data = features_sorted[:,features_list[1:].index(feature)]
    
    for person_index, feature_value in enumerate(feature_data):
#        person_indexName = data_dict.keys()[np.where(indexes_sorted==person_index)[0]]
        person_indexName = names[indexes_sorted[person_index]]
        if (feature_value > 7) or (feature_value < -7):
            feature_unscaled = int(np.array(features)[:,features_list[1:].index(feature)][indexes_sorted[person_index]])
            print '{0:28s} {1:10f}  {2:10d}  {3:6d}   {4:30s}'.format(feature, feature_value, feature_unscaled, person_index, person_indexName)
#            print '{0:28s} {1:10f}  {2:6d}   {3:30s}'.format(feature, feature_value, person_index, person_indexName)

    plt.plot(data_range, feature_data, color=clr)
plt.plot(data_range, scale(labels_sorted), color='k', linewidth=3)
plt.show()

#for feature in features_list[1:]:
#    clr = [c,(c + c0) % 1,(c + 2*c0) % 1]
#    c = (c + 5*c0) % 1
#    feature_data = features_sorted[:,features_list[1:].index(feature)]
#    
#    plt.plot(data_range, feature_data, color=clr)
#    plt.plot(data_range, scale(labels_sorted), color='k', linewidth=3)
#    plt.title(feature)
#    plt.show()
    
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
data = featureFormat(my_dataset, features_list, sort_keys = True, remove_NaN=False)
labels, features = targetFeatureSplit(data)
features_scaled = scale(features)

for feature in features_list[:0:-1]:   
    feature_index = features_list[1:].index(feature)
    feature_data = features_scaled[:,feature_index]
    for person_index, feature_value in enumerate(feature_data):
        if math.isnan(feature_value):
            features_scaled[person_index][feature_index] = 0

'''
        NEW FEATURES !
        use the words in the content of the emails.
        find words that are good indicator for poi and define it as a feature.
        this can ce done buy text inspection.
        or we can just use all the words and create a bag of words
        like in the sara and mashmo exersize.
        then do a massive dimention reduction by vectorising and pca.

'''


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




print            
sys.exit(0)

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