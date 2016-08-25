# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 11:28:29 2016

@author: Xin
"""

import pandas as pd
#import randomforests as rf
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics as skm
import pylab as pl

samsungdata = pd.read_csv('samsungdata.csv')
samtrain = pd.read_csv('samtrain.csv')
samtest = pd.read_csv('samtest.csv')
samval = pd.read_csv('samval.csv')
#samsungmin = pd.read_csv('samsungmin.csv')

#samtrain = rf.remap_col(samtrain,'activity')
#samval = rf.remap_col(samval,'activity')
#samtest = rf.remap_col(samtest,'activity')

X_train = samtrain.iloc[:,1:36]
y_train = samtrain['activity']

X_test = samtest.iloc[:,1:36]
y_test = samtest['activity']

X_val = samval.iloc[:,1:36]
y_val = samval['activity']

rfc = RandomForestClassifier(n_estimators=500, oob_score=True)
model = rfc.fit(X_train, y_train)
print("OOB (out of bag) score = %f" % rfc.oob_score_)

score_test = rfc.score(X_test, y_test)
score_val = rfc.score(X_val, y_val)

print("mean accuracy score for validation set = %f" % score_val)
print("mean accuracy score for test set = %f" % score_test)

#feature selection
fi = enumerate(rfc.feature_importances_)
cols = X_train.columns
[(value, cols[i]) for (i, value) in fi if value > 0.04]

# plot the confusion matrix 
test_pred = rfc.predict(X_test)
cm = skm.confusion_matrix(y_test, test_pred)

pl.matshow(cm)
pl.title('Confusion matrix for test data')
pl.colorbar()
pl.show()

# compute precision and recall score 
# Precision
print("Precision = %f" %(skm.precision_score(y_test, test_pred, average='weighted')))
# Recall
print("Recall = %f" %(skm.recall_score(y_test, test_pred, average='weighted')))


### Full random forests analysis
# make a list with a new set of column names
cols = list(samsungdata.columns)
newcols = [ "x%d"%(k) for k in range(0,len(cols)) ] 
newcols[-2:] = cols[-2:] 
samsungdata.columns = newcols
samsungdata.columns[0]


full_train = samsungdata[samsungdata['subject'] >= 27]
full_test = samsungdata[samsungdata['subject'] <= 6]
full_val2 = samsungdata[samsungdata['subject'] < 27]
full_val = full_val2[full_val2['subject'] >= 21 ]
X_ftrain = full_train.iloc[:,1:-2]
y_ftrain = full_train['activity']
X_ftest = full_test.iloc[:,1:-2]
y_ftest = full_test['activity']
X_fval= full_val.iloc[:,1:-2]
y_fval = full_val['activity']

rfc_full = RandomForestClassifier(n_estimators=500, oob_score=True)
model_full = rfc_full.fit(X_ftrain, y_ftrain)
print("OOB (out of bag) score = %f" % rfc_full.oob_score_)

score_test_full = rfc_full.score(X_ftest, y_ftest)
score_val_full = rfc_full.score(X_fval, y_fval)

print("mean accuracy score for validation set = %f" % score_val_full)
print("mean accuracy score for test set = %f" % score_test_full)

#feature selection
fi_full = enumerate(rfc_full.feature_importances_)
cols = X_ftrain.columns
[(value, cols[i]) for (i, value) in fi_full if value > 0.015]

# plot the confusion matrix 
test_pred_full = rfc_full.predict(X_ftest)
# Precision
print("Precision = %f" %(skm.precision_score(y_ftest, test_pred_full, average='weighted')))
# Recall
print("Recall = %f" %(skm.recall_score(y_ftest, test_pred_full, average='weighted')))

fi_full = enumerate(rfc_full.feature_importances_)
cols = X_ftrain.columns
top10 = [(cols[i]) for (i,value) in fi_full if value > 0.015]
origindx = [int(x[1:]) for x in top10]
samsungdata = pd.read_csv('samsungdata.csv')
samsungdata.columns[origindx]