'''
author: Lama Hamadeh
'''

import pandas as pd
import matplotlib.pyplot as plt


#
# TODO: Load up the Seeds Dataset into a Dataframe
# It's located at 'Datasets/wheat.data'
# 
# .. your code here ..

wheat_dataset=pd.read_csv('/Users/ADB3HAMADL/Desktop/Anaconda_Packages/DAT210x-master/Module3/Datasets/wheat.data',index_col = 0)
#
# TODO: Drop the 'id' feature
# 
# .. your code here ..


#
# TODO: Compute the correlation matrix of your dataframe
# 
# .. your code here ..

wheat_dataset=wheat_dataset.corr() #creating the correlation matrix for our dataframe
#
# TODO: Graph the correlation matrix using imshow or matshow
# 
# .. your code here ..
plt.imshow(wheat_dataset.corr(), cmap=plt.cm.Blues, interpolation='nearest')
plt.colorbar()
tick_marks = [i for i in range(len(wheat_dataset.columns))]
plt.xticks(tick_marks, wheat_dataset.columns, rotation='vertical')
plt.yticks(tick_marks, wheat_dataset.columns)

plt.show()


print(wheat_dataset.corr()) #printing the correlation matrix for our dataframe
