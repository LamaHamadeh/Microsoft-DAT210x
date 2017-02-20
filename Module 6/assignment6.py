# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 11:22:46 2017

@author: Lama Hamadeh
"""

import pandas as pd
import time

# Grab the DLA HAR dataset from:
# http://groupware.les.inf.puc-rio.br/har
# http://groupware.les.inf.puc-rio.br/static/har/dataset-har-PUC-Rio-ugulino.zip


#
# TODO: Load up the dataset into dataframe 'X'
# TODO: Clean up any column with commas in it
# so that they're properly represented as decimals instead
#
# .. your code here ..

X = pd.read_csv('/Users/ADB3HAMADL/Desktop/dataset-har-PUC-Rio-ugulino/dataset-har-PUC-Rio-ugulino.csv', sep = ';', decimal = ',')

print(X.shape) #165633x19

#
# TODO: Encode the gender column, 0 as male, 1 as female
#
# .. your code here ..

X.gender = X.gender.map({'Man': 0, 'Woman': 1})
#print (X)


#
# INFO: Check data types
print (X.dtypes)


#
# TODO: Convert any column that needs to be converted into numeric
# use errors='raise'. This will alert you if something ends up being
# problematic
#
# .. your code here ..

X.z4 = pd.to_numeric(X.z4, errors = 'coerce') #z4 is the only coordinate that has an 'object' type rather than a 'numeric/float' one.
print (X.dtypes)


#
# INFO: If you find any problematic records, drop them before calling the
# to_numeric methods above...

# let's check for Nans values

# INFO: An easy way to show which rows have nans in them
#print X[pd.isnull(X).any(axis=1)]

#OR

#identify nans
def num_missing(x):
  return sum(x.isnull())
#Applying per column:
print ("Missing values per column:")
print (X.apply(num_missing, axis=0)) #axis=0 defines that function is to be applied on each column

#z4 has only one Nan value

X.dropna(axis = 0, how = 'any', inplace = True)    
      
print(list(X.columns.values)) #print the columns headers of the dataset to know which column to copy

#
# TODO: Encode your 'y' value as a dummies version/copy version of your dataset's "class" column
#
# .. your code here ..

y = X['class'].copy()

#
# TODO: Get rid of the user and class columns
#
# .. your code here ..

X.drop(['user','class'], axis = 1, inplace = True)

print (X.describe())

'''

#
# TODO: Create an RForest classifier 'model' and set n_estimators=30,
# the max_depth to 10, and oob_score=True, and random_state=0
#
# .. your code here ..



# 
# TODO: Split your data into test / train sets
# Your test size can be 30% with random_state 7
# Use variable names: X_train, X_test, y_train, y_test
#
# .. your code here ..





print "Fitting..."
s = time.time()
#
# TODO: train your model on your training set
#
# .. your code here ..
print "Fitting completed in: ", time.time() - s


#
# INFO: Display the OOB Score of your data
score = model.oob_score_
print "OOB Score: ", round(score*100, 3)




print "Scoring..."
s = time.time()
#
# TODO: score your model on your test set
#
# .. your code here ..
print "Score: ", round(score*100, 3)
print "Scoring completed in: ", time.time() - s


#
# TODO: Answer the lab questions, then come back to experiment more


#
# TODO: Try playing around with the gender column
# Encode it as Male:1, Female:0
# Try encoding it to pandas dummies
# Also try dropping it. See how it affects the score
# This will be a key on how features affect your overall scoring
# and why it's important to choose good ones.



#
# TODO: After that, try messing with 'y'. Right now its encoded with
# dummies try other encoding methods to experiment with the effect.
'''