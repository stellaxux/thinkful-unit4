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

data['class'].replace('Iris-setosa',1, inplace = True)
data['class'].replace('Iris-versicolor', 2, inplace = True)
data['class'].replace('Iris-virginica', 3, inplace = True)

plt.scatter(data.sepal_length, data.sepal_width, c=data['class'])
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.show()

length = random.uniform(4, 8)
width = random.uniform(1, 5)

# self-written knn
for index, row in data.iterrows():
    data.loc[index, 'distance'] = math.sqrt((row['sepal_length']-length)**2 + (row['sepal_width']-width)**2)
nearest = data.nsmallest(10, 'distance')
nearest['class'].value_counts().idxmax()
#'Iris-versicolor'


# apply KNN from sklearn
knn = KNeighborsClassifier(10)
X = data[[0,1]].values
y = data['class'].as_matrix()
predicted = knn.fit(X,y).predict(X)
accuracy = knn.score(X, y)

test = np.asarray([length, width])
test = test.reshape(1, 2) 
print knn.predict(test)
#'Iris-versicolor'