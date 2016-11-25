# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 14:23:43 2016

@author: ADB3HAMADL
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import load_iris
from pandas.tools.plotting import parallel_coordinates
from pandas.tools.plotting import andrews_curves

plt.style.use('ggplot') 


student_dataset=pd.read_csv('/Users/ADB3HAMADL/Desktop/Anaconda_Packages/students.data', index_col=0)

#print(student_dataset)



'''
#Histograms
my_series = student_dataset.G3

my_dataframe = student_dataset[['G3', 'G2', 'G1']] 

my_series.plot.hist(bins=20, alpha=0.5, normed=True)

my_dataframe.plot.hist(bins=20, alpha=0.5, normed=True)
plt.show()
'''
#---------------------------------------------------------------------------------------------------------------

'''
#2DScatter plots
student_dataset.plot.scatter(x='G1', y='G3')
plt.show()
'''
'''
#3DScatter plots
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('Final Grade')
ax.set_ylabel('First Grade')
ax.set_zlabel('Daily Alcohol')

ax.scatter(student_dataset.G1, student_dataset.G3, student_dataset['Dalc'], c='r', marker='.')
plt.show()
'''
#---------------------------------------------------------------------------------------------------------------

'''
#Parllel Coordinates
# Load up SKLearn's Iris Dataset into a Pandas Dataframe
data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names) 

df['target_names'] = [data.target_names[i] for i in data.target]

# Parallel Coordinates Start Here:
plt.figure()
parallel_coordinates(df, 'target_names')
plt.show()
'''

'''
Pandas' parallel coordinates interface is extremely easy to use, but use it with care. It only supports a single scale for all your axes. If you have some features that are on a small scale and others on a large scale, you'll have to deal with a compressed plot. For now, your only three options are to:

Normalize your features before charting them
Change the scale to a log scale
Or create separate, multiple parallel coordinate charts. Each one only plotting features with similar domains scales plotted
'''
#---------------------------------------------------------------------------------------------------------------

'''
#Andrew's Curve
# Load up SKLearn's Iris Dataset into a Pandas Dataframe
data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)

df['target_names'] = [data.target_names[i] for i in data.target]

# Andrews Curves Start Here:
plt.figure()
andrews_curves(df, 'target_names')
plt.show()
#---------------------------------------------------------------------------------------------------------------
'''
'''
#Imshaow
df = pd.DataFrame(np.random.randn(1000, 5), columns=['a', 'b', 'c', 'd', 'e'])
print( df.corr())
plt.imshow(df.corr(), cmap=plt.cm.Blues, interpolation='nearest')
plt.colorbar()
tick_marks = [i for i in range(len(df.columns))]
plt.xticks(tick_marks, df.columns, rotation='vertical')
plt.yticks(tick_marks, df.columns)
plt.show()
#---------------------------------------------------------------------------------------------------------------
'''

'''
x = np.arange(0, 10,0.1);  #np.arange (start,stop,step) on x axis
y = np.sin(x)
plt.scatter(x, y)
plt.show()
'''
#---------------------------------------------------------------------------------------------------------------
