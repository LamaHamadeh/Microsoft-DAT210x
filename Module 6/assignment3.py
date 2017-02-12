#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 21:34:08 2017

@author: lamahamadeh
"""

import pandas as pd

#TODO Load up the /Module6/Datasets/parkinsons.data data set into a variable X, 
#being sure to drop the name column.

X = pd.read_csv('/Users/lamahamadeh/Downloads/Modules/DAT210x-master/Module6/Datasets/parkinsons.data')
X.drop('name', axis = 1, inplace = True)

print (X.head())
print (X.describe())

#checking nans values
def num_missing(x):
  return sum(x.isnull())

#Applying per column:
print ("Missing values per column:")
print (X.apply(num_missing, axis=0)) #No Nans in the dataset

#checking the type of the data      
print (X.dtypes) 


#TODO Splice out the status column into a variable y and delete it from X.

y = X['status'].copy()
X.drop('status', axis = 1, inplace = True)

#TODO Perform a train/test split. 30% test group size, with a random_state equal to 7.

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 7)


#TODO Create a SVC classifier. Don't specify any parameters, just leave everything as default. 
#Fit it against your training data and then score your testing data.

from sklearn.svm import SVC
model = SVC()
model.fit(X_train, y_train)
score = model.score(X_test, y_test)
print(score)



