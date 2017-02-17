'''
author Lama Hamadeh
'''

import pandas as pd


#https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.names


# 
# TODO: Load up the mushroom dataset into dataframe 'X'
# Verify you did it properly.
# Indices shouldn't be doubled.
# Header information is on the dataset's website at the UCI ML Repo
# Check NA Encoding
#
# .. your code here ..

X = pd.read_csv('/Users/Admin/Desktop/LAMA/DAT210x/DAT210x-master/Module6/Datasets/agaricus-lepiota.data', na_values = '?')
#we use na_values='?' here to prevent pandas from interpreting 'NA' or 'N' as NaN (and vice versa) in a string and define 
#the exact NANs in the dataset especially when your dataset contains string values

X.columns = ['classification', 'cap-shape', 'cap-surface', 'cap-color', 'bruises?', 'odor', 'gill-attachment', 
                'gill-spacing', 'gill-size', 'gill-color', 'stalk-shape', 'stalk-root', 'stalk-surface-above-ring', 
                'stalk-surface-below-ring', 'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type', 'veil-color', 
                'ring-number', 'ring-type', 'spore-print-color', 'population', 'habitat']

print(X.head(5))

print "Before dropping all rows with any NaNs, shape of X is:", X.shape

# INFO: An easy way to show which rows have nans in them
#print X[pd.isnull(X).any(axis=1)]

#OR

#check which column have nans values in order o apply .fillna
def num_missing(x):
  return sum(x.isnull())

#Applying per column:
print "Missing values per column:"
print X.apply(num_missing, axis=0) #axis=0 defines that function is to be applied on each column

# TODO: Go ahead and drop any row with a nan
#
# .. your code here ..
X.dropna(axis = 0, how = 'any', inplace = True)

print "After dropping all rows with any NaNs, shape of X is:", X.shape


#
# TODO: Copy the labels out of the dset into variable 'y' then Remove
# them from X. Encode the labels, using the .map() trick we showed
# you in Module 5 -- canadian:0, kama:1, and rosa:2
#
# .. your code here ..

y = X['classification'].copy()
X.drop('classification', axis = 1, inplace = True)
y = y.map({'e': 0, 'p': 1})

#
# TODO: Encode the entire dataset using dummies
#
# .. your code here ..

X = pd.get_dummies(X, columns = ['cap-shape', 'cap-surface', 'cap-color', 'bruises?', 'odor', 'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color', 'stalk-shape', 'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring', 'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number', 'ring-type', 'spore-print-color', 'population', 'habitat'])
# 
# TODO: Split your data into test / train sets
# Your test size can be 30% with random_state 7
# Use variable names: X_train, X_test, y_train, y_test
#
# .. your code here ..

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 7)

#
# TODO: Create an DT classifier. No need to set any parameters
#
# .. your code here ..

from sklearn import tree
DT = tree.DecisionTreeClassifier()

#
# TODO: train the classifier on the training data / labels:
# TODO: score the classifier on the testing data / labels:
#
# .. your code here ..

DT.fit(X_train, y_train)
score = DT.score(X_test, y_test)
print "High-Dimensionality Score: ", round((score*100), 3)


#
# TODO: Use the code on the courses SciKit-Learn page to output a .DOT file
# Then render the .DOT to .PNGs. Ensure you have graphviz installed.
# If not, `brew install graphviz. If you can't, use: http://webgraphviz.com/
#
# .. your code here ..

tree.export_graphviz(DT, out_file = 'tree.dot', feature_names = X.columns)

import subprocess
subprocess.call(['dot', '-Tpdf', 'tree.dot', '-o' 'tree.pdf'])

     
