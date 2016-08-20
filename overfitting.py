# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 21:13:20 2016

@author: Xin
"""

import numpy as np
import statsmodels.formula.api as smf
import pandas as pd

# set seed for reproducible results
np.random.seed(414)

# get toy data
X = np.linspace(0, 15, 1000)
y = 3* np.sin(X) + np.random.normal(1+X, 0.2, 1000)

train_X, train_y = X[:700], y[:700]
test_X, test_y = X[700:], y[700:]

train_df = pd.DataFrame({'X': train_X, 'y': train_y})
test_df = pd.DataFrame({'X': test_X, 'y':test_y})

#linear fit
poly_1 = smf.ols(formula='y ~ 1 + X', data=train_df).fit()
poly_1.summary()
#training set mean squared error
print ("Linear model: mean squared error of training set: %.2f" % np.mean((poly_1.predict() - train_y) ** 2))

#testing set mean squared error
print("Linear model: mean squared error of testing set: %.2f" % np.mean((poly_1.predict(test_df) - test_y) ** 2))

#quadratic fit
poly_2 = smf.ols(formula='y ~ 1 + X + I(X**2)', data=train_df).fit()
poly_2.summary()
#training set mean squared error
print("Quadratic model: mean squared error of training set: %.2f" % np.mean((poly_2.predict() - train_y) ** 2))

#testing set mean squared error
print("Quadratic model: mean squared error of testing set: %.2f" % np.mean((poly_2.predict(test_df) - test_y) ** 2))

