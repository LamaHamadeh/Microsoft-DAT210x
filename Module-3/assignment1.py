'''
author: Lama Hamadeh
'''


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

# Look pretty...
matplotlib.style.use('ggplot')

#
# TODO: Load up the Seeds Dataset into a Dataframe
# It's located at 'Datasets/wheat.data'
# 
# .. your code here ..
wheat_dataset=pd.read_csv('/Users/ADB3HAMADL/Desktop/Anaconda_Packages/DAT210x-master/Module3/Datasets/wheat.data', index_col=0)
print(wheat_dataset)
#
# TODO: Create a slice of your dataframe (call it s1)
# that only includes the 'area' and 'perimeter' features
# 
# .. your code here ..

slice_1=wheat_dataset.loc[:,['area','perimeter']]
print(slice_1)



	
#slice_1= wheat_dataset[['area', 'perimeter']] 

#
# TODO: Create another slice of your dataframe (call it s2)
# that only includes the 'groove' and 'asymmetry' features
# 
# .. your code here ..


slice_2=wheat_dataset.loc[:,['groove','asymmetry']]
print(slice_2)


#
# TODO: Create a histogram plot using the first slice,
# and another histogram plot using the second slice.
# Be sure to set alpha=0.75
# 
# .. your code here ..


slice_1.plot.hist(alpha=0.75)
slice_2.plot.hist(alpha=0.75)
plt.show()

