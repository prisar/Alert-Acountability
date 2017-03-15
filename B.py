import time
t0 = time.time()

from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import random as rd
import pandas as pd
import numpy as np
import xgboost as xgb
import nltk

dataset_train = pd.read_csv('train_merged.csv', delimiter=",")
dataset_test = pd.read_csv('test_merged.csv', delimiter=",")

from sklearn.base import TransformerMixin
class DataFrameImputer(TransformerMixin):
    def fit(self, X, y=None):
        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].median() for c in X],
            index=X.columns)
        return self
    def transform(self, X, y=None):
        return X.fillna(self.fill)

feature_columns_to_use = [ 'serviceType', 'callCountIdentifier', 'site', 'DayofWeek ', 'deviceCategory', 'application', 'functionalArea', 'networkCategory', 'maker']
non_numerical_cols = ['serviceType', 'callCountIdentifier', 'site', 'DayofWeek ', 'deviceCategory', 'application', 'functionalArea', 'networkCategory', 'maker']

big_X = dataset_train[feature_columns_to_use].append(dataset_test[feature_columns_to_use])
big_X_imputed = DataFrameImputer().fit_transform(big_X)


label_encoder = LabelEncoder()
for feature in non_numerical_cols:
  big_X_imputed[feature] = label_encoder.fit_transform(big_X_imputed[feature])

train_X = big_X_imputed[0:dataset_train.shape[0]].as_matrix()
test_X = big_X_imputed[dataset_train.shape[0]::].as_matrix()
print(dataset_train.columns.tolist())
train_y = dataset_train['Actionability']
print(train_X)

'''
alg = LogisticRegression(random_state=1)

alg.fit(big_X_imputed[feature_columns_to_use], big_X_imputed['Actionability'])

prediction = alg.predict(dataset_test[feature_columns_to_use])
'''

gbm = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05).fit(train_X, train_y)
predictions = gbm.predict(test_X)

submission = pd.DataFrame({ 'serialNum': dataset_test['serialNum'],
  'Actionability': predictions
  })

sub = submission[['serialNum', 'Actionability']]
sub.to_csv('submit.csv', index=False)

print("Time elapsed: %.2f sec" % (time.time() - t0))