import pandas as pd
import html5lib

# TODO: Load up the table, and extract the dataset
# out of it. If you're having issues with this, look
# carefully at the sample code provided in the reading
#
# .. your code here ..
NHL_DataFrame = pd.read_html('http://www.espn.com/nhl/statistics/player/_/stat/points/sort/points/year/2015/seasontype/2')[0]
#print (NHL_DataFrame)

# TODO: Rename the columns so that they match the
# column definitions provided to you on the website
#
# .. your code here ..
NHL_DataFrame.columns = ['RK','PLAYER','TEAM','GP','SMG','SMA','PTS','plusminus','PIM','PTSG','SOG','PCT','GWG','SMG','SMA','SMG','SMA']
print (NHL_DataFrame)


# TODO: Get rid of any row that has at least 4 NANs in it
#
# .. your code here ..
NHL_DataFrame = NHL_DataFrame.dropna(axis=0, thresh=4)
print (NHL_DataFrame)
print (NHL_DataFrame.shape)


# TODO: At this point, look through your dataset by printing
# it. There probably still are some erroneous rows in there.
# What indexing command(s) can you use to select all rows
# EXCEPT those rows?
#
# .. your code here ..

NHL_DataFrame.my_feature.fillna( NHL_DataFrame.my_feature.mean() )

# TODO: Get rid of the 'RK' column
#
# .. your code here ..
NHL_DataFrame = NHL_DataFrame.drop(labels=['RK'], axis=1)
print (NHL_DataFrame)

# TODO: Ensure there are no holes in your index by resetting
# it. By the way, don't store the original index
#
# .. your code here ..

NHL_DataFrame = NHL_DataFrame.reset_index(drop=True)

# TODO: Check the data type of all columns, and ensure those
# that should be numeric are numeric
print(NHL_DataFrame.dtypes)

#NHL_DataFrame.GP = pd.to_numeric(NHL_DataFrame.GP, errors='coerce')
#NHL_DataFrame.SMG = pd.to_numeric(NHL_DataFrame.SMG, errors='coerce')
#NHL_DataFrame.SMA = pd.to_numeric(NHL_DataFrame.SMA, errors='coerce')
#NHL_DataFrame.PTS = pd.to_numeric(NHL_DataFrame.PTS, errors='coerce')
#NHL_DataFrame.plusminus = pd.to_numeric(NHL_DataFrame.plusminus, errors='coerce')
#NHL_DataFrame.PIM = pd.to_numeric(NHL_DataFrame.PIM, errors='coerce')
#NHL_DataFrame.PTSG = pd.to_numeric(NHL_DataFrame.PTSG, errors='coerce')
#NHL_DataFrame.SOG = pd.to_numeric(NHL_DataFrame.SOG, errors='coerce')
#NHL_DataFrame.PCT = pd.to_numeric(NHL_DataFrame.PCT, errors='coerce')
#NHL_DataFrame.GWG = pd.to_numeric(NHL_DataFrame.GWG, errors='coerce')

#NHL_DataFrame.fillna(0)
#NHL_DataFrame_cleaned = NHL_DataFrame.dropna(how='all')
#print(NHL_DataFrame.PCT.unique())
#print (NHL_DataFrame)
#print(NHL_DataFrame.dtypes)
# TODO: Your dataframe is now ready! Use the appropriate 
# commands to answer the questions on the course lab page.

