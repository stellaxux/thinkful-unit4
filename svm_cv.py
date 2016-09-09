# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 22:10:44 2016

@author: Xin
"""

from sklearn import datasets
from sklearn import svm
from sklearn import cross_validation

iris = datasets.load_iris()

svc = svm.SVC(kernel='linear', C=1)
X = iris.data[0:150, 0:4]
y = iris.target[0:150]
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.4, stratify=y)
model = svc.fit(X_train, y_train)
score = svc.score(X_test, y_test)

score_cv = cross_validation.cross_val_score(svc, X_train, y_train, cv=5)
print("Mean accuracy: %0.2f" % score_cv.mean())
print("Std of accuracy: %0.2f" % score_cv.std())

f1 = cross_validation.cross_val_score(svc, X_train, y_train, cv=5, scoring='f1_macro') # very close to accuracy
precision = cross_validation.cross_val_score(svc, X_train, y_train, cv=5, scoring='precision_macro')
recall = cross_validation.cross_val_score(svc, X_train, y_train, cv=5, scoring='recall_macro')