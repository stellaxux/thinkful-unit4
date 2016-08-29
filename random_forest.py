# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 11:28:29 2016

@author: Xin
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics as skm
import pylab as pl

#######  random forest analysis with domain knowledge (38 feature selected) ############

#reduced samsung data with 38 features selected by domain knowledge
sammin = pd.read_csv('samsungmin.csv') # sammin.shape (7352, 38) 

# split reduced data into train, test and validation
samtrain = sammin[sammin['subject'] >= 27]
samtest = sammin[sammin['subject'] <= 6]
samval2 = sammin[sammin['subject'] < 27]
samval = samval2[samval2['subject'] >= 21 ]

X_train = samtrain.iloc[:,1:-2]
y_train = samtrain['activity']

X_test = samtest.iloc[:,1:-2]
y_test = samtest['activity']

X_val = samval.iloc[:,1:-2]
y_val = samval['activity']

# fit model using training set and calculate accuracy score on training set
rfc = RandomForestClassifier(n_estimators=500, oob_score=True)
model = rfc.fit(X_train, y_train)
print("OOB (out of bag) score = %f" % rfc.oob_score_) # OOB (out of bag) score = 0.985185

# accuracy score on test and validation set
score_test = rfc.score(X_test, y_test)
score_val = rfc.score(X_val, y_val)
print("mean accuracy score for validation set = %f" % score_val) # mean accuracy score for validation set = 0.757098
print("mean accuracy score for test set = %f" % score_test) # mean accuracy score for test set = 0.809886

#feature selection, top 10
fi = enumerate(rfc.feature_importances_)
cols = X_train.columns
[(value, cols[i]) for (i, value) in fi if value > 0.04]
#[(0.042945116807908551, 'tAccMean'),
# (0.050109274767225459, 'tAccStd'),
# (0.04537773341594814, 'tJerkMean'),
# (0.043382734472001194, 'tGyroJerkMean'),
# (0.051858489750216605, 'fAccMean'),
# (0.045986770702916721, 'fAccSD'),
# (0.16182971003157759, 'angleXGravity'),
# (0.11814404435142693, 'angleYGravity'),
# (0.10108468766061068, 'angleZGravity')]


# plot the confusion matrix 
test_pred = rfc.predict(X_test)
cm = skm.confusion_matrix(y_test, test_pred)

pl.matshow(cm)
pl.title('Confusion matrix for test data')
pl.colorbar()
pl.show()

# Compute precision and recall score 
# Precision
print("Precision = %f" %(skm.precision_score(y_test, test_pred, average='weighted'))) # 0.818673
# Recall
print("Recall = %f" %(skm.recall_score(y_test, test_pred, average='weighted'))) # 0.809886



############ Full random forests analysis with all features included #################

#full samsung data with 564 features
samsungdata = pd.read_csv('samsungdata.csv') # samsungdata.shape (7352, 564)

# change column names to x1, x2,...  
cols = list(samsungdata.columns)
newcols = [ "x%d"%(k) for k in range(0,len(cols)) ] 
newcols[-2:] = cols[-2:] 
samsungdata.columns = newcols
samsungdata.columns[0]

# split data into train, test and validation
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

# fit model using training set and calculate accuracy score on training set
rfc_full = RandomForestClassifier(n_estimators=500, oob_score=True)
model_full = rfc_full.fit(X_ftrain, y_ftrain)
print("OOB (out of bag) score = %f" % rfc_full.oob_score_) # OOB (out of bag) score = 0.992593

# accuracy score on test and validation set
score_test_full = rfc_full.score(X_ftest, y_ftest)
score_val_full = rfc_full.score(X_fval, y_fval)
print("mean accuracy score for validation set = %f" % score_val_full) # 0.792850
print("mean accuracy score for test set = %f" % score_test_full) # 0.825856

#feature selection
fi_full = enumerate(rfc_full.feature_importances_)
cols = X_ftrain.columns
[(value, cols[i]) for (i, value) in fi_full if value > 0.015]

# Compute precision and recall score 
test_pred_full = rfc_full.predict(X_ftest)
# Precision
print("Precision = %f" %(skm.precision_score(y_ftest, test_pred_full, average='weighted'))) # 0.840633
# Recall
print("Recall = %f" %(skm.recall_score(y_ftest, test_pred_full, average='weighted'))) # 0.825856

#feature selection, top 10
fi_full = enumerate(rfc_full.feature_importances_)
cols = X_ftrain.columns
top10 = [(cols[i]) for (i,value) in fi_full if value > 0.015]
origindx = [int(x[1:]) for x in top10]
samsungdata = pd.read_csv('samsungdata.csv')
samsungdata.columns[origindx]
#      [u'tGravityAcc-mean()-X', u'tGravityAcc-mean()-Y',
#       u'tGravityAcc-max()-X', u'tGravityAcc-max()-Z', u'tGravityAcc-min()-X',
#       u'tGravityAcc-min()-Y', u'tGravityAcc-min()-Z',
#       u'tGravityAcc-energy()-X', u'tGravityAcc-energy()-Z',
#       u'angle(X,gravityMean)', u'angle(Z,gravityMean)']