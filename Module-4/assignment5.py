import pandas as pd

from scipy import misc
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import matplotlib.pyplot as plt

# Look pretty...
matplotlib.style.use('ggplot')


#
# TODO: Start by creating a regular old, plain, "vanilla"
# python list. You can call it 'samples'.
#
# .. your code here .. 

samples= []

#
# TODO: Write a for-loop that iterates over the images in the
# Module4/Datasets/ALOI/32/ folder, appending each of them to
# your list. Each .PNG image should first be loaded into a
# temporary NDArray, just as shown in the Feature
# Representation reading.
#
# Optional: Resample the image down by a factor of two if you
# have a slower computer. You can also convert the image from
# 0-255  to  0.0-1.0  if you'd like, but that will have no
# effect on the algorithm's results.
#
# .. your code here .. 

import os

for ALOI32 in os.listdir('/Users/ADB3HAMADL/Desktop/Anaconda_Packages/DAT210x-master/Module4/Datasets/ALOI/32'):
	a = os.path.join('Datasets/ALOI/32', ALOI32)
	img1 = misc.imread(a).reshape(-1)
	samples.append(img1)
print len(samples)


#
# TODO: Once you're done answering the first three questions,
# right before you converted your list to a dataframe, add in
# additional code which also appends to your list the images
# in the Module4/Datasets/ALOI/32_i directory. Re-run your
# assignment and answer the final question below.
#
# .. your code here .. 
for ALOI32i in os.listdir('/Users/ADB3HAMADL/Desktop/Anaconda_Packages/DAT210x-master/Module4/Datasets/ALOI/32i'):
	b = os.path.join('Datasets/ALOI/32i', ALOI32i)
	img2 = misc.imread(b).reshape(-1)
	samples.append(img2)
print len(samples)


#create a colur python list where.
#Store a 'b' in it for each element you load from the /32/ directory, 
#and an 'r' for each element you load from the '32_i' directory. 
#Then pass this variable to your 2D and 3D scatter plots, as an optional parameter c=colors
# .. your code here .. 



#
# TODO: Convert the list to a dataframe
#
# .. your code here .. 

df = pd.DataFrame(samples)
#
# TODO: Implement Isomap here. Reduce the dataframe df down
# to three components, using K=6 for your neighborhood size
#
# .. your code here .. 

from sklearn import manifold
iso = manifold.Isomap(n_neighbors=1, n_components=3)
T=iso.fit_transform(df)
manifold.Isomap(eigen_solver='auto', max_iter=None, n_components=3, n_neighbors=1,
    neighbors_algorithm='auto', path_method='auto', tol=0)

#def Plot3D(T, title, x, y, num_to_plot=40):
def Plot2D(T, title, x, y, num_to_plot=40):    
    
#
# TODO: Create a 2D Scatter plot to graph your manifold. You
# can use either 'o' or '.' as your marker. Graph the first two
# isomap components
#
# .. your code here .. 

  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.set_title(title)
  ax.set_xlabel('Component: {0}'.format(x))
  ax.set_ylabel('Component: {0}'.format(y))
  x_size = (max(T[:,x]) - min(T[:,x])) * 0.08
  y_size = (max(T[:,y]) - min(T[:,y])) * 0.08
  ax.scatter(T[:,x],T[:,y], marker='.',alpha=0.7)

Plot2D(T, 'Isomap 2D', 0, 1, num_to_plot=40)

#
# TODO: Create a 3D Scatter plot to graph your manifold. You
# can use either 'o' or '.' as your marker:
#
# .. your code here .. 

def Plot3D(T, title, x, y, z):
  fig = plt.figure()
  ax = fig.add_subplot(111, projection = '3d')
  ax.set_title(title)
  ax.set_xlabel('Component: {0}'.format(x))
  ax.set_ylabel('Component: {0}'.format(y))
  ax.set_zlabel('Component: {0}'.format(z))
  x_size = (max(T[:,x]) - min(T[:,x])) * 0.08
  y_size = (max(T[:,y]) - min(T[:,y])) * 0.08
  z_size = (max(T[:,z]) - min(T[:,z])) * 0.08
  ax.scatter(T[:,x],T[:,y],T[:,z], marker='.', alpha=0.7)

Plot3D(T, "Isomap 3D", 0, 1, 2)

plt.show()

