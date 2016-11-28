import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from pandas.tools.plotting import parallel_coordinates

# Look pretty...
matplotlib.style.use('ggplot')


#
# TODO: Load up the Seeds Dataset into a Dataframe
# It's located at 'Datasets/wheat.data'
# 
# .. your code here ..

wheat_dataset=pd.read_csv('/Users/ADB3HAMADL/Desktop/Anaconda_Packages/DAT210x-master/Module3/Datasets/wheat.data',index_col = 0)
#print(wheat_dataset)
#
# TODO: Drop the 'id', 'area', and 'perimeter' feature
# 
# .. your code here .

wheat_dataset = wheat_dataset.reset_index(drop=True) #deleting 'id' column

New_wheat_dataset=wheat_dataset.drop(['area', 'perimeter'], axis=1) #deleting columns by labels

#New_wheat_dataset=wheat_dataset.drop(wheat_dataset.columns[[0,1]], axis=1) #deleting columns by indeices
#
# TODO: Plot a parallel coordinates chart grouped by
# the 'wheat_type' feature. Be sure to set the optional
# display parameter alpha to 0.4
# 
# .. your code here ..


# Load up SKLearn's Iris Dataset into a Pandas Dataframe
data = load_iris()

# Parallel Coordinates Start Here:
plt.figure()
parallel_coordinates(New_wheat_dataset, 'wheat_type',alpha= 0.4)
plt.show()


