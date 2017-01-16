# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 11:43:18 2016

@author: ADB3HAMADL 
"""

'''
Data wrangling information listed here has been taken mostly from Chris Albon's website: chrisAlbon.com
'''


#Numpy Array Basics
import numpy as np
# Create a list
battle_deaths = [3246, 326, 2754, 2547, 2457, 3456]
print(battle_deaths)
# Create an array from numpy
deaths = np.array(battle_deaths)
print(deaths)
# Create an array of zeros
defectors = np.zeros(6)
print(defectors)
# Create a range from 0 to 100
zero_to_99 = np.arange(0, 100)
print(zero_to_99)
# Create 100 ticks between 0 and 1
zero_to_1 = np.linspace(0, 1, 100)
print(zero_to_1)
# Mean value of the array
civilian_deaths = np.array([4352, 233, 3245, 256, 2394])
civilian_deaths.mean()
# Total amount of deaths
civilian_deaths.sum()
# Smallest value in the array
civilian_deaths.min()
# Largest value in the array
civilian_deaths.max()
#------------------------------------------------------------------------------


#Indexing And Slicing Numpy Arrays
import numpy as np
# Create a 2x2 array
battle_deaths = [[344, 2345], [253, 4345]]
deaths = np.array(battle_deaths)
print(battle_deaths)
print(deaths)
# Select the top row, second item
print(deaths[0, 1])
# Select the second column
print(deaths[:, 1])
# Select the second row
print(deaths[1, :])
#------------------------------------------------------------------------------


#Pandas Dataframes
import pandas as pd
import numpy as np

#Creating Dataframe
data = {'name': ['Jason', 'Molly', 'Tina', 'Jake', 'Amy'], 
        'year': [2012, 2012, 2013, 2014, 2014], 
        'reports': [4, 24, 31, 2, 3],
        'coverage': [25, 94, 57, 62, 70]} #python arange the columns alphabetically
df = pd.DataFrame(data, index = ['Cochice', 'Pima', 'Santa Cruz', 'Maricopa', 'Yuma'])#without the index there will be an index column starting from 0
#print(df) 

#Applying A Function Over A Dataframe
#Create a function that multiplies all non-strings by 100
# create a function called times100
def times100(x):
    # that, if x is a string,
    if type(x) is str:
        # just returns it untouched
        return x
    # but, if not, return it multiplied by 100
    elif x:
        return 100 * x
    # and leave everything else
    else:
        return
        
df=df.applymap(times100)        
print(df) 

# Replace the dataframe with a new one which does not contain the first row
df = df[1:]

#Transpose the dataframe
df=df.T

#Saving A Pandas Dataframe As A CSV
#Safe the dataframe called "df" as csv

#To see a descriptive statistical summary of your dataframe's numeric columns
df.describe()
#----------------


#Rows
#drop a single row
#df = df.drop('Cochice', axis=0)

#drop multiple rows
df=df.drop(['Cochice', 'Pima'], axis=0)

#Drop a row if it contains a certain value (in this case, "Tina")
df = df[df.name != 'Tina']

#View the first two rows of the dataframe
df=df[:2]

#View all rows where coverage is more than 50
df=df[df['coverage'] > 50]

#View a row
df.ix['Maricopa']

#Select all rows by index label
df.loc[:'Cochice']

#Sort the dataframe's rows by reports, in descending order
df.sort_values(by='reports', ascending=0)

#Sort the dataframe's rows by coverage and then by reports, in ascending order
df.sort_values(by=['coverage', 'reports'])

#Select rows by row number
df.iloc[:2]# Select every row up to 3
df.iloc[1:2]# Select the second and third row
df.iloc[2:]# Select every row after the third row

#----------------
#Columns
#drop a single column
df = df.drop('name', axis=1)

#Add a column
df['working hours'] =[12, 13, 14, 15, 10]

#rename columns headers
df=df.columns = ['new', 'column', 'header', 'labels','New']

#View two columns of the dataframe
df=df[['name', 'reports']]

#display the name of the columns in your dataframe
 df.columns
 

#View a column
df.ix[:, 'coverage']

#Select a column
df['coverage']

#Select multiple columns
df[['coverage', 'name']]

#Select rows by column number
df.iloc[:,:2]# Select the first 2 columns



print(df) 
#----------------
#matplotlib and plot cusomisation
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 10, 20)#we can use logspace instead of linespace but we have to be careful as the starting and the ending point must 
#be the logarithmic values of the original start and end points.
y1 = x**2.0
y2 = x**1.5
plt.plot(x, y1, 'bo-', linewidth = 2, markersize = 12, label = 'First')#instead of plot we can use loglog function where the values on
#x axis and y axis are the logarithmic values.
plt.plot(x, y2, 'gs-', linewidth = 2, markersize = 12, label = 'Second')
plt.xlabel('$X$') #matplotlibe support LaTex typsetting 
plt.ylabel('$Y$')
plt.axis([-0.5, 10.5, -5, 105])#organise the axes
plt.legend(loc = 'upper left')#create a legend based on the labels specified earlier for each plot and define its position 
plt.savefig('myplot.pdf')#sae the figure as a .pdf file (you can save it as .png file as well).
#------------------------------------------------------------------------------



