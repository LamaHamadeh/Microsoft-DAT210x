import pandas as pd
import numpy as np


#
# TODO:
# Load up the dataset, setting correct header labels.
#
# .. your code here ..
Census_DataFrame = pd.read_csv('/Users/Admin/Desktop/DAT210x/DAT210x-master/Module2/Datasets/census.data',na_values=["?"])
Census_DataFrame = Census_DataFrame.drop(labels=['0'], axis=1)
Census_DataFrame.columns=['education', 'age', 'capital-gain', 'race', 'capital-loss', 'hours-per-week', 'sex', 'classification']
Census_DataFrame = Census_DataFrame.fillna(0)
#print(Census_DataFrame.dtypes)




#ordered_age = ['20', '25', '30', '35','40', '45', '50', '55', '60']
#Census_DataFrame.age = Census_DataFrame.age.astype("category", ordered=True, categories=ordered_age).cat.codes

ordered_education = ['5th', '6th', '7th', '8th','7th-8th', '9th', '10th', '11th', '12th', 'HS-grad', 'Some-college' , 'Bachelors','Masters','Doctorate' ]
Census_DataFrame.education = Census_DataFrame.education.astype("category", ordered=True, categories=ordered_education).cat.codes
print(Census_DataFrame)

#
# TODO:
# Use basic pandas commands to look through the dataset... get a
# feel for it before proceeding! Do the data-types of each column
# reflect the values you see when you look through the data using
# a text editor / spread sheet program? If you see 'object' where
# you expect to see 'int32' / 'float64', that is a good indicator
# that there is probably a string or missing value in a column.
# use `your_data_frame['your_column'].unique()` to see the unique
# values of each column and identify the rogue values. If these
# should be represented as nans, you can convert them using
# na_values when loading the dataframe.
#
# .. your code here ..



#
# TODO:
# Look through your data and identify any potential categorical
# features. Ensure you properly encode any ordinal and nominal
# types using the methods discussed in the chapter.
#
# Be careful! Some features can be represented as either categorical
# or continuous (numerical). Think to yourself, does it generally
# make more sense to have a numeric type or a series of categories
# for these somewhat ambigious features?
#
# .. your code here ..



#
# TODO:
# Print out your dataframe
#
# .. your code here ..


