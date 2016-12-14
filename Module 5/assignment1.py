#
# TOOD: Import whatever needs to be imported to make this work
#
# .. your code here ..

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans



matplotlib.style.use('ggplot') # Look Pretty


#
# TODO: To procure the dataset, follow these steps:
# 1. Navigate to: https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-present/ijzp-q8t2
# 2. In the 'Primary Type' column, click on the 'Menu' button next to the info button,
#    and select 'Filter This Column'. It might take a second for the filter option to
#    show up, since it has to load the entire list first.
# 3. Scroll down to 'GAMBLING'
# 4. Click the light blue 'Export' button next to the 'Filter' button, and select 'Download As CSV'

#
# TODO: Load your dataset after importing Pandas
#
# .. your code here ..

df=pd.read_csv('/Users/lamahamadeh/Downloads/Modules/DAT210x-master/Module5/Datasets/Crimes_-_2001_to_present.csv', index_col=0)


#
# TODO: Drop any ROWs with nans in them
#
# .. your code here ..

df= df.dropna(axis = 0, how = 'any')
print (df)
#
# TODO: Print out the dtypes of your dset
#
# .. your code here ..

print(df.dtypes)
#
# Coerce the 'Date' feature (which is currently a string object) into real date,
# and confirm by re-printing the dtypes. NOTE: This is a slow process...
#
# .. your code here ..

df.Date = pd.to_datetime(df.Date) # Converts the entries in the 'Date' column to datetime64[ns]
print (df.dtypes)

#
# TODO: Filter df so that you're only looking at Longitude and Latitude,
# since the remaining columns aren't really applicable for this purpose.
#
# .. your code here ..

df1=df[['Longitude','Latitude']]

print (df1)




#
# INFO: Plot your data with a '.' marker, with 0.3 alpha at the Longitude,
# and Latitude locations in your dataset. Longitude = x, Latitude = y
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.scatter(df.Longitude, df.Longitude, marker='.', alpha=0.3)
plt.show()
  

  #
  # TODO: Use K-Means to try and find seven cluster centers in this df.
  #
  # .. your code here ..
  
def doKMeans(df):
    kmeans_model = KMeans(n_clusters=7)
    kmeans_model.fit(df)
    KMeans(copy_x=True, init='k-means++', max_iter=300, n_clusters=7, n_init=10,
    n_jobs=1, precompute_distances='auto', random_state=None, tol=0.0001,
    verbose=0)
    labels = kmeans_model.predict(df)
    
  # INFO: Print and plot the centroids...
  
    centroids = kmeans_model.cluster_centers_
    ax.scatter(centroids[:,0], centroids[:,1], marker='x', c='red', alpha=0.5, linewidths=3, s=169)
    print (centroids)
 

# INFO: Print & Plot your data
doKMeans(df)


#
# TODO: Filter out the data so that it only contains samples that have
# a Date > '2011-01-01', using indexing. Then, in a new figure, plot the
# crime incidents, as well as a new K-Means run's centroids.
#
# .. your code here ..

df1 = df[df.Date > '2011-01-01']

# INFO: Print & Plot your data
doKMeans(df1)
plt.title("Dates limited to 2011 and later")

plt.show()
