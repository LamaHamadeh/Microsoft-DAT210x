# -*- coding: utf-8 -*-
"""
Spyder Editor

Authot: Lama Hamadeh 28/11/2016

This file is to create a 6-D parallel coordinate chart for x=[0, 2pi] and the relations with its sin, cos, tan, exp, sqrt and sqruare values.

*Still working on it!*
"""

import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pandas.tools.plotting import parallel_coordinates
matplotlib.style.use('ggplot')



x=np.arange(0,math.pi*2,math.pi/40)
#print(x)
#print(math.pi)


data = {'-x':x, 
        'x':-x,
        'Sin(x)':np.sin(x),
        'Cos(x)': np.cos(x),
        'Tan(x)': np.tan(x),
        'exp(x)': np.exp(x),
        'Square(x)': x*x,
        'Sqrt(x)': np.sqrt(x)} #python arange the columns alphabetically
df = pd.DataFrame(data, index=None) #without the index there will be an index column starting from 0
print(df) 

'''
plt.scatter(x,Sinx)
plt.show()
'''

plt.figure()
parallel_coordinates(df,'x',alpha= 0.4)
plt.show()
