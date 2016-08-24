# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 20:42:50 2016

@author: Xin
"""

from sklearn import cross_validation
from sklearn import linear_model
import pandas as pd

data = pd.read_csv('loansData_clean.csv')

data['constant'] = 1.0

X = ['constant', 'FICO.Score', 'Amount.Requested']

lr = linear_model.LinearRegression()

mae = cross_validation.cross_val_score(lr, data[X], data['Interest.Rate'], cv=10, scoring='mean_absolute_error')
print("MAE: %0.2f" % mae.mean())

mse = cross_validation.cross_val_score(lr, data[X], data['Interest.Rate'], cv=10, scoring='mean_squared_error')
print("MSE: %0.2f" % mse.mean())

r2 = cross_validation.cross_val_score(lr, data[X], data['Interest.Rate'], cv=10)

print("R-squared: %0.2f" % r2.mean())

#MAE: -0.02
#MSE: -0.00
#R-squared: 0.65
