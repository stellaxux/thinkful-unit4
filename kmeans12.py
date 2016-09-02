# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 14:10:19 2016

@author: Xin
"""

import pandas as pd
from scipy.cluster.vq import vq, kmeans, whiten, kmeans2
import matplotlib.pyplot as plt


data = pd.read_csv('un.csv')
len(data) #207

# count the number of non-null values present in each column
data.count()
# determine the data type of each column.
data.dtypes
# numbers of countries present in the dataset
len(data['country'].unique())

# fill NAN with 0
data = data.fillna(0)
lifemale = data['lifeMale']
lifefemale = data['lifeFemale']
gdp = data['GDPperCapita']
infantmortality = data['infantMortality']


# cluster lifemale on gdp
d1 = {'gdp': gdp, 'lifemale': lifemale}
df1 = pd.DataFrame(d1)
cluster = df1.values

# Normalize on a per feature basis
cluster1 = whiten(cluster)

# determine the ideal number of clusters
avg_distance = []
for k in range(1,10):
    centroids1,dist1 = kmeans(cluster1,k)
    idx1,idxdist1 = vq(cluster1,centroids1)    
  # avg_dist = np.mean(idxdist1) same as dist1
    avg_distance.append(dist1)
    
# Just plotting the mean distance, you can plot Euclidian distance once you update the code
plt.figure()
plt.plot(range(1,10), avg_distance)
plt.show()

# kmeans cluster, k=3, determined from the above plot
centroids1,dist1 = kmeans(cluster1,3)
# assign obs to centroids and calculate distance between obs and its centroids
idx1,idxdist1 = vq(cluster1,centroids1)
# plot the clusters 
plt.figure()
plt.plot(cluster[idx1==0,0],cluster[idx1==0,1],'ob',
     cluster[idx1==1,0],cluster[idx1==1,1],'or',
     cluster[idx1==2,0],cluster[idx1==2,1],'og')
plt.xlabel('Per Capita GDP in USD')
plt.ylabel('Male life expectancy')
plt.show()


# cluster infantmortality on gdp
d1 = {'gdp': gdp, 'infantmortality': infantmortality}
df1 = pd.DataFrame(d1)
cluster = df1.values
cluster1 = whiten(cluster)

# kmeans cluster, k=3, determined from the above plot
centroids1,dist1 = kmeans(cluster1,3)
# assign obs to centroids and calculate distance between obs and its centroids
idx1,idxdist1 = vq(cluster1,centroids1)

# plot the clusters 
plt.figure()
plt.plot(cluster[idx1==0,0],cluster[idx1==0,1],'ob',
     cluster[idx1==1,0],cluster[idx1==1,1],'or',
     cluster[idx1==2,0],cluster[idx1==2,1],'og')
plt.xlabel('Per Capita GDP in USD')
plt.ylabel('Infant mortality')
plt.show()

# cluster infantmortality on gdp using kmeans2
d1 = {'gdp': gdp, 'infantmortality': infantmortality}
df1 = pd.DataFrame(d1)
cluster = df1.values
cluster1 = whiten(cluster)

# kmeans cluster, k=3, determined from the above plot
centroids1,labl = kmeans2(cluster1,3)
# assign obs to centroids and calculate distance between obs and its centroids
# idx1,idxdist1 = vq(cluster1,centroids1)

# plot the clusters 
plt.figure()
plt.plot(cluster[labl==0,0],cluster[labl==0,1],'ob',
     cluster[labl==1,0],cluster[labl==1,1],'or',
     cluster[labl==2,0],cluster[labl==2,1],'og')
plt.xlabel('Per Capita GDP in USD')
plt.ylabel('Infant mortality')
plt.show()