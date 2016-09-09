# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 09:38:40 2016

@author: Xin
"""

import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier


cols = ['sepal_length','sepal_width','petal_length','petal_width','class']
data = pd.read_csv('iris.data.csv', names=cols)


data['class'].replace('Iris-setosa',1, inplace = True)
data['class'].replace('Iris-versicolor', 2, inplace = True)
data['class'].replace('Iris-virginica', 3, inplace = True)

X = data[[0,1,2,3]].values
y = data['class'].values

# LDA
sklearn_lda = LDA(n_components=2)
X_lda = sklearn_lda.fit_transform(X, y)

# plot the LDA
label_dict = {1: 'Setosa', 2: 'Versicolor', 3:'Virginica'}
for label,marker,color in zip(np.unique(y),('^', 's', 'o'),('blue', 'red', 'green')):
    plt.scatter(X_lda[y==label, 0], X_lda[y==label, 1]*-1,color=color, marker=marker,label=label_dict[label])
plt.xlabel('LD1')
plt.ylabel('LD2')
plt.legend(loc='lower center', fancybox=True)
plt.title('LDA: Iris projection onto the first 2 linear discriminants')

# perform KNN with lda data
knn_lda = KNeighborsClassifier(n_neighbors=6)
predicted_lda = knn_lda.fit(X_lda,y).predict(X_lda)
accuracy_lda = knn_lda.score(X_lda, y)