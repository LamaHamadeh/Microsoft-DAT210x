import random, math
import pandas as pd
import numpy as np
import scipy.io

from mpl_toolkits.mplot3d import Axes3D

from sklearn.decomposition import PCA
from sklearn import manifold

from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import matplotlib

#------------------------------------
# If you'd like to try this lab with PCA instead of Isomap,
# as the dimensionality reduction technique:
Test_PCA = False


def plotDecisionBoundary(model, X, y):
  print "Plotting..."
  matplotlib.style.use('ggplot') # Look Pretty

  fig = plt.figure()
  ax = fig.add_subplot(111)

  padding = 0.1
  resolution = 0.1

  #(2 for benign, 4 for malignant)
  colors = {2:'royalblue',4:'lightsalmon'} 

  
  # Calculate the boundaris
  x_min, x_max = X[:, 0].min(), X[:, 0].max()
  y_min, y_max = X[:, 1].min(), X[:, 1].max()
  x_range = x_max - x_min
  y_range = y_max - y_min
  x_min -= x_range * padding
  y_min -= y_range * padding
  x_max += x_range * padding
  y_max += y_range * padding

  # Create a 2D Grid Matrix. The values stored in the matrix
  # are the predictions of the class at at said location
  xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution),
                       np.arange(y_min, y_max, resolution))

  # What class does the classifier say?
  Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
  Z = Z.reshape(xx.shape)

  # Plot the contour map
  plt.contourf(xx, yy, Z, cmap=plt.cm.seismic)
  plt.axis('tight')

  # Plot your testing points as well...
  for label in np.unique(y):
    indices = np.where(y == label)
    plt.scatter(X[indices, 0], X[indices, 1], c=colors[label], alpha=0.8)

  p = model.get_params()
  plt.title('K = ' + str(p['n_neighbors']))
  plt.show()

#------------------------------------
# 
# TODO: Load in the dataset, identify nans, and set proper headers.
# Be sure to verify the rows line up by looking at the file in a text editor.
#
# .. your code here ..

#loading dataframe
X = pd.read_csv('/Users/Admin/Desktop/DAT210x/DAT210x-master/Module5/Datasets/breast-cancer-wisconsin.data')

#renaming columns
X.columns = ['sample', 'thickness', 'size', 'shape', 'adhesion', 'epithelial', 'nuclei', 'chromatin', 'nucleoli', 'mitoses', 'status']

#print the first 7 raws of the dataframe
print(X.head(5))
print(X.dtypes)
#it can be seen that the 'nuclei' column has an 'object' type where it has to be 'int'. Therefore, we need to change it to numeric value
X.nuclei = pd.to_numeric(X.nuclei, errors = 'coerce')
print(X.dtypes)

#identify nans
def num_missing(x):
  return sum(x.isnull())
#Applying per column:
print "Missing values per column:"
print X.apply(num_missing, axis=0) #axis=0 defines that function is to be applied on each column
#'nuclei' column has 16 nans.
#------------------------------------

# 
# TODO: Copy out the status column into a slice, then drop it from the main
# dataframe. You can also drop the sample column, since that doesn't provide
# us with any machine learning power.
#
# .. your code here ..

y = X['status'].copy()

X.drop(labels = ['sample', 'status'], inplace = True, axis = 1)

#------------------------------------

#
# TODO: With the labels safely extracted from the dataset, replace any nan values
# with the mean feature / column value
#
# .. your code here ..
X.nuclei.fillna(X.nuclei.mean(), inplace = True)
#X = X.fillna(method = 'backfill', axis = 1)

#------------------------------------

#
# TODO: Experiment with the basic SKLearn preprocessing scalers. We know that
# the features consist of different units mixed in together, so it might be
# reasonable to assume feature scaling is necessary. Print out a description
# of the dataset, post transformation.
#
# .. your code here ..

from sklearn import preprocessing
T = X #Having no normalising to the data values at all

#We can use also MaxAbsScaler(), MinMaxScaler(), StandardScaler(), Normalizer(), RobustScaler() as other preprocessing methods
#IMPORTANT NOTE
#Just because you know _how_ to perform preprocessing scaling doesn't mandate it become a required step for every machine 
#learning dataset. In the breast cancer dataset, from the description, it seems like all the data is already standardized 
#from 1-10 (ints), so no further preprocessing is required. In fact, attempting to do so further brings down the predictive accuracy!
#therefore, we still can use non of these methods and we get a correct approximation at the end.
#------------------------------------
#
# PCA and Isomap are your new best friends
model = None
if Test_PCA:
  print "Computing 2D Principle Components"
  #
  # TODO: Implement PCA here. save your model into the variable 'model'.
  # You should reduce down to two dimensions.
  #
  # .. your code here ..

  model = PCA(n_components = 2)

else:
  print "Computing 2D Isomap Manifold"
  #
  # TODO: Implement Isomap here. save your model into the variable 'model'
  # Experiment with K values from 5-10.
  # You should reduce down to two dimensions.
  #
  # .. your code here ..
  model = manifold.Isomap(n_neighbors = 6, n_components = 2)

#
# TODO: Train your model against data_train, then transform both
# data_train and data_test using your model. You can save the results right
# back into the variables themselves.
#
# .. your code here ..

X = model.fit_transform(T)
#------------------------------------

#
# TODO: Do train_test_split. Use the same variable names as on the EdX platform in
# the reading material, but set the random_state=7 for reproduceability, and keep
# the test_size at 0.5 (50%).
#
# .. your code here ..

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, random_state = 7)


#------------------------------------
# 
# TODO: Implement and train KNeighborsClassifier on your projected 2D
# training data here. You can use any K value from 1 - 15, so play around
# with it and see what results you can come up. Your goal is to find a
# good balance where you aren't too specific (low-K), nor are you too
# general (high-K). You should also experiment with how changing the weights
# parameter affects the results.
#
# .. your code here ..

from sklearn.neighbors import KNeighborsClassifier
knmodel = KNeighborsClassifier(n_neighbors = 2, weights = 'distance')
knmodel.fit(X_train, y_train)
#------------------------------------
#
# INFO: Be sure to always keep the domain of the problem in mind! It's
# WAY more important to errantly classify a benign tumor as malignant,
# and have it removed, than to incorrectly leave a malignant tumor, believing
# it to be benign, and then having the patient progress in cancer. Since the UDF
# weights don't give you any class information, the only way to introduce this
# data into SKLearn's KNN Classifier is by "baking" it into your data. For
# example, randomly reducing the ratio of benign samples compared to malignant
# samples from the training set.

#
# TODO: Calculate + Print the accuracy of the testing set
#
# .. your code here ..

print knmodel.score(X_test, y_test)
#------------------------------------


plotDecisionBoundary(knmodel, X_test, y_test)

