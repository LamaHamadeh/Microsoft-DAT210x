
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 21:34:08 2017
@author: lamahamadeh
"""

#TODO Load up the /Module6/Datasets/parkinsons.data data set into a variable X, 
#being sure to drop the name column.

import pandas as pd

X = pd.read_csv('/Users/Admin/Desktop/LAMA/DAT210x/DAT210x-master/Module6/Datasets/parkinsons.data')
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
print(score) #0.813559322034

#-----------------------------------------------------------------------------------------

#That accuracy was just too low to be useful. We need to get it up. 
#One way you could go about doing that would be to manually try a bunch of 
#combinations of C, and gamma values for your rbf kernel. 
#But that could literally take forever. Also, you might unknowingly skip 
#a pair of values that would have resulted in a very good accuracy.

#TODO Instead, lets get the computer to do what computers do best. 
#Program a naive, best-parameter search by creating nested for-loops. 
#The outer for-loop should iterate a variable C from 0.05 to 2, using 0.05 unit increments. 
#The inner for-loop should increment a variable gamma from 0.001 to 0.1, using 0.001 unit increments. 
#As you know, Python ranges won't allow for float intervals, so you'll have to do some research on NumPy ARanges, 
#if you don't already know how to use them.

#TODO Since the goal is to find the parameters that result in the model having the best accuracy score, 
#you'll need a best_score = 0 variable that you initialize outside of the for-loops. Inside the inner for-loop, 
#create an SVC model and pass in the C and gamma parameters its class constructor. Train and score the model appropriately. 
#If the current best_score is less than the model's score, update the best_score being sure to print it out, 
#along with the C and gamma values that resulted in it.

import numpy as np

''' 
=====
NOTE:
=====
# for float interval we can use np.arange as following:
for i in np.arange(start,stop,step):
    print i
#-------------------------------------
 
#or we can define afunction that does the same thing as:
def MakeRange(start,stop,step):
    i = 0
    while start + i * step < stop:
        yield start + i * step
        i += 1
                
for i in MakeRange(start,stop,step):
    print i #the results are numbers not a list
'''

best_score = 0

for i in np.arange(start = 0.05, stop = 2, step = 0.05): #variable C from 0.05 to 2, using 0.05 unit increments
    for j in np.arange(start = 0.001, stop = 0.1, step = 0.001): #Tvariable gamma from 0.001 to 0.1, using 0.001 unit increments
        #creating a SVC model
        model = SVC(C = i, gamma = j)
        #training the data
        model.fit(X_train, y_train)
        # TODO: Calculate the score of your SVC against the testing data
        score = model.score(X_test, y_test)
        if score > best_score:
            best_score = score
            best_C = model.C
            best_gamma = model.gamma
print ("The highest score obtained:", best_score) #0.915254237288
print ("C value:", best_C) #1.65
print ("gamma value:", best_gamma) #0.005

#-----------------------------------------------------------------------------------------

#Wait a second. Pull open the dataset's label file from: https://archive.ics.uci.edu/ml/datasets/Parkinsons

#Look at the units on those columns: Hz, %, Abs, dB, etc. What happened to transforming your data? 
#With all of those units interacting with one another, some pre-processing is surely in order.

#Right after you preform the train/test split but before you train your model, inject SciKit-Learn's pre-processing code. 
#Unless you have a good idea which one is going to work best, you're going to have to try the various pre-processors one at a time, 
#checking to see if they improve your predictive accuracy.

#Experiment with Normalizer(), MaxAbsScaler(), MinMaxScaler(), KernelCenterer(), and StandardScaler().

from sklearn import preprocessing

T = preprocessing.Normalizer().fit_transform(X)
#T = preprocessing.MaxAbsScaler().fit_transform(X)
#T = preprocessing.MinMaxScaler().fit_transform(X)
#T = preprocessing.KernelCenterer().fit_transform(X)
#T = preprocessing.StandardScaler().fit_transform(X)
#T = X # No Change

#-----------------------------------------------------------------------------------------








#-----------------------------------------------------------------------------------------
