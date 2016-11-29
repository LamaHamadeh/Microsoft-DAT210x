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



x=np.arange(0,np.pi*2,np.pi/40)
#print(x)
#print(math.pi)


data = {'x':x, 
        '-x':-x,
        'Sin(x)':np.sin(x),
        'Cos(x)': np.cos(x),
        'Tan(x)': np.tan(x),
        'exp(x)': np.exp(x),
        'Square(x)': x*x,
        'Sqrt(x)': np.sqrt(x)} #python arange the columns alphabetically
df = pd.DataFrame(data, index=None) #without the index there will be an index column starting from 0
print(df) 

#Playing with plots and subplots before going to parallel coordinates
plt.figure(1)
plt.subplot(311)
plt.plot(x,np.sin(x),'b', linewidth=2.0)
plt.xlabel('x')
plt.ylabel('Sin(x)')
plt.grid(False)
plt.xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi],
           ['$0$', r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$', r'$2\pi$'])

plt.subplot(312)
plt.plot(x,np.cos(x),'r', linewidth=2.0)
plt.xlabel('x')
plt.ylabel('Cos(x)')
plt.grid(False)
plt.xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi],
           ['$0$', r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$', r'$2\pi$'])

plt.subplot(313)
plt.plot(x,np.tan(x),'g', linewidth=2.0)
plt.xlabel('x')
plt.ylabel('Tan(x)')
plt.grid(False)
plt.xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi],
           ['$0$', r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$', r'$2\pi$'])

plt.show()


plt.plot(x,np.sin(x),'b',x,np.cos(x),'r', linewidth=2.0)
plt.xlabel('x')
plt.ylabel('Cos(x) (in red) and Sin(x) (in blue)')
plt.xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi],
           ['$0$', r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$', r'$2\pi$'])

plt.show()

#Here we go!
plt.figure()
parallel_coordinates(df,'x',alpha= 0.4)
plt.show()
