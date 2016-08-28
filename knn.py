# -*- coding: utf-8 -*-
"""
Created on Sat Aug 27 15:42:32 2016

@author: Xin
"""

import pandas as pd
import matplotlib.pyplot as plt
import random
import math
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

cols = ['sepal_length','sepal_width','petal_length','petal_width','class']
data = pd.read_csv('iris.data.csv', names=cols)

groups = data.groupby('class')
colors = {'Iris-setosa':'red', 'Iris-versicolor':'blue', 'Iris-virginica':'green'}

for name, group in groups:
    plt.scatter(group.sepal_length, group.sepal_width, c=colors[name], label=name)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.legend()
plt.show()

length = random.uniform(4, 8)
width = random.uniform(1, 5)

for index, row in data.iterrows():
    data.loc[index, 'distance'] = math.sqrt((row['sepal_length']-length)**2 + (row['sepal_width']-width)**2)
nearest = data.nsmallest(10, 'distance')
nearest['class'].value_counts().idxmax()

# apply KNN from sklearn
knn = KNeighborsClassifier(10)
X = data[['sepal_length','sepal_width']].as_matrix()
y = data['class'].as_matrix()
knn.fit(X,y)

test = np.asarray([length, width])
test = test.reshape(1, 2) 
print knn.predict(test)
#'Iris-versicolor'