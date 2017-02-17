'''
author Lama Hamadeh
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from sklearn import preprocessing
from sklearn.decomposition import PCA

matplotlib.style.use('ggplot') # Look Pretty
#------------------------------------

def plotDecisionBoundary(model, X, y):
  fig = plt.figure()
  ax = fig.add_subplot(111)

  padding = 0.6
  resolution = 0.0025
  colors = ['royalblue','forestgreen','ghostwhite']

  # Calculate the boundaries
  x_min, x_max = X[:, 0].min(), X[:, 0].max()
  y_min, y_max = X[:, 1].min(), X[:, 1].max()
  x_range = x_max - x_min
  y_range = y_max - y_min
  x_min -= x_range * padding
  y_min -= y_range * padding
  x_max += x_range * padding
  y_max += y_range * padding

  # Create a 2D Grid Matrix. The values stored in the matrix are the predictions of the class at at said location
  xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution), np.arange(y_min, y_max, resolution))

  # What class does the classifier say?
  Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
  Z = Z.reshape(xx.shape)

  # Plot the contour map
  cs = plt.contourf(xx, yy, Z, cmap=plt.cm.terrain)

  # Plot the test original points as well...
  for label in range(len(np.unique(y))):
    indices = np.where(y == label)
    plt.scatter(X[indices, 0], X[indices, 1], c=colors[label], label=str(label), alpha=0.8)

  p = model.get_params()
  plt.axis('tight')
  plt.title('K = ' + str(p['n_neighbors']))

#------------------------------------
# 
# TODO: Load up the dataset into a variable called X. Check the .head and
# compare it to the file you loaded in a text editor. Make sure you're
# loading your data properly--don't fail on the 1st step!
#
# .. your code here ..

X = pd.read_csv('/Users/Admin/Desktop/DAT210x/DAT210x-master/Module5/Datasets/wheat.data')
#print X.head()
#------------------------------------
#
# TODO: Copy the 'wheat_type' series slice out of X, and into a series
# called 'y'. Then drop the original 'wheat_type' and the 'id' columns from the X
#
# .. your code here ..

y=X['wheat_type'].copy()

X.drop(labels = ['id', 'wheat_type'], inplace = True, axis = 1)
#------------------------------------

# TODO: Do a quick, "ordinal" conversion of 'y'. In actuality our
# classification isn't ordinal, but just as an experiment...
#
# .. your code here ..

y = y.astype('category').cat.codes
#------------------------------------
#
# TODO: Basic nan munging. Fill each row's nans with the mean of the feature
#
# .. your code here ..

#check which column have nans values in order o apply .fillna
def num_missing(x):
  return sum(x.isnull())

#Applying per column:
print "Missing values per column:"
print X.apply(num_missing, axis=0) #axis=0 defines that function is to be applied on each column

#As the result indicats that 'compactness', 'width' and 'groove' have nan values where
#nans of compactness = 3
#nans of width = 1
#nans of groove = 4
#then we can apply the .fillna on them.

X.compactness.fillna(X.compactness.mean(), inplace = True)
X.width.fillna(X.width.mean(), inplace = True)
X.groove.fillna(X.groove.mean(), inplace = True)

#------------------------------------
#
# TODO: Just like your preprocessing transformation, create a PCA
# transformation as well. Fit it against your training data, and then
# project your training and testing features into PCA space using the
# PCA model's .transform() method.
#
# NOTE: This has to be done because the only way to visualize the decision
# boundary in 2D would be if your KNN algo ran in 2D as well:
#
# .. your code here ..

T = preprocessing.normalize(X)
pca = PCA(n_components = 2)
pca_X = pca.fit_transform(T)


#------------------------------------
#
# TODO: Split X into training and testing data sets using train_test_split().
# INFO: Use 0.33 test size, and use random_state=1. This is important
# so that your answers are verifiable. In the real world, you wouldn't
# specify a random_state.
#
# .. your code here ..

from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(pca_X, y, test_size = 0.33, random_state = 1)
#We use X_train and y_train for training (fit), and use X_test and y_test for predicting or scoring.
##------------------------------------ 

#
# TODO: Create and train a KNeighborsClassifier. Start with K=9 neighbors.
# NOTE: Be sure train your classifier against the pre-processed, PCA-
# transformed training data above! You do not, of course, need to transform
# your labels.
#
# .. your code here ..

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 9)
knn.fit(X_train, y_train)


# HINT: Ensure your KNeighbors classifier object from earlier is called 'knn'
plotDecisionBoundary(knn, X_train, y_train)


#------------------------------------
#
# TODO: Display the accuracy score of your test data/labels, computed by
# your KNeighbors model.
#
# NOTE: You do NOT have to run .predict before calling .score, since
# .score will take care of running your predictions for you automatically.
#
# .. your code here ..

print knn.score(X_test, y_test)
#------------------------------------
#
# BONUS: Instead of the ordinal conversion, try and get this assignment
# working with a proper Pandas get_dummies for feature encoding. HINT:
# You might have to update some of the plotDecisionBoundary code.


plt.show()
