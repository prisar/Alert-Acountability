import time
t0 = time.time()

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV as cc

train = pd.read_csv('temp.csv')
test = pd.read_csv('test_simple.csv')
print (train.shape)
print (test.shape)
test_ids = test.serialNum

levels = train.species
print (levels)

'''
train.drop(['creationDate', 'serialNum'], axis=1, inplace=True)
test.drop(['serialNum'], axis=1, inplace=True)
print ("after")
print (levels.shape)
print (train.shape)
print (test.shape)
print ("number of classes = ", levels.unique().shape)

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder().fit(levels)
levels=le.transform(levels)

Model=RandomForestClassifier(n_estimators=1000)
Model = cc(Model, cv=3, method='isotonic')
Model.fit(train, levels)

predictions = Model.predict_proba(test)
print (predictions.shape)
sub = pd.DataFrame(predictions, columns=list(le.classes_))
sub.insert(0, 'id', test_ids)
sub.reset_index()
sub.to_csv('submit.csv', index = False)
# print (sub.head())
'''

print("Time elapsed: %.2f sec" % (time.time() - t0))