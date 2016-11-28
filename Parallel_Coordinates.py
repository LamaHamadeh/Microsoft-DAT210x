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



x=np.arange(0,math.pi*2,0.1)
#print(x)
#print(math.pi)


Sinx=pd.Series(np.sin(x))
#print(Sinx)

Cosx=pd.Series(np.cos(x))
#print(Cosx)

Tanx=pd.Series(np.tan(x))
#print(Tanx)

Expx=pd.Series(np.exp(x))
#print(Expx)

Squarex=pd.Series(x**2)
#print(Squarex)

Sqrtx=pd.Series(np.sqrt(x))
#print(Sqrtx)


data = {'-x':[x], 
        'x':[-x],
        'Sin(x)': [Sinx],
        'Cos(x)': [Cosx],
        'Tan(x)': [Tanx],
        'exp(x)': [Expx],
        'Square(x)': [Squarex],
        'Sqrt(x)': [Sqrtx]} #python arange the columns alphabetically
df = pd.DataFrame(data, index=None) #without the index there will be an index column starting from 0
#print(df) 

'''
plt.scatter(x,Sinx)
plt.show()
'''

plt.figure()
parallel_coordinates(df,'x',alpha= 0.4)
plt.show()
