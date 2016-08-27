# -*- coding: utf-8 -*-
"""
Created on Sat Aug 27 11:05:01 2016

@author: Xin
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB

data = pd.read_csv('ideal_weight.csv')

cols = list(data.columns)
cols = [x.replace("'", "") for x in cols]

data.columns = cols

data['sex'] = [x.replace("'", "") for x in data['sex']]

plt.hist(data['ideal'],alpha=0.5,bins=25,label='Ideal')
plt.hist(data['actual'],alpha=0.5,bins=25,label='Actual')
plt.legend()
plt.show()

plt.hist(data['diff'],alpha=0.5,bins=25)
plt.show()

data['gender'] = pd.Categorical(list(data['sex']))
data['gender'].value_counts() # Female 119, Male 63

gnb = GaussianNB()
X = data[['actual','ideal','diff']]
y = data['gender']
model = gnb.fit(X, y)
predicted = model.predict(X)
print("Number of mislabeled points out of a total %d points: %d" %(len(X), (y != predicted).sum()))
# Number of mislabeled points out of a total 182 points: 14

d = {'actual': 145, 'ideal': 160, 'diff': -15}
df = pd.DataFrame(data=d, index=[1])
# df = df[['actual','ideal', 'diff']]
pred = model.predict(df)
print pred