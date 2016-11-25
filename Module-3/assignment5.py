#
# This code is intentionally missing!
# Read the directions on the course lab page!
#

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from pandas.tools.plotting import parallel_coordinates

# Look pretty...
matplotlib.style.use('ggplot')

wheat_dataset=pd.read_csv('/Users/ADB3HAMADL/Desktop/Anaconda_Packages/DAT210x-master/Module3/Datasets/wheat.data',index_col = 0)
'''
New_wheat_dataset=wheat_dataset.drop(['area', 'perimeter'], axis=1) #deleting columns by labels

data = load_iris()

# Parallel Coordinates Start Here:
plt.figure()
andrews_curves(New_wheat_dataset, 'wheat_type',alpha= 0.4)
plt.show()
'''

matplotlib.style.use('ggplot')

wheat_dataset=pd.read_csv('/Users/ADB3HAMADL/Desktop/Anaconda_Packages/DAT210x-master/Module3/Datasets/wheat.data',index_col = 0)

data = load_iris()

# Parallel Coordinates Start Here:
plt.figure()
andrews_curves(wheat_dataset, 'wheat_type',alpha= 0.4)
plt.show()