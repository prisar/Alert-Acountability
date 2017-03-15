import time
t0 = time.time()

import pandas as pd
import numpy as np
import re
import nltk

train_df = pd.read_csv('./train_merged.csv', delimiter=',')
train_df.shape
train_df.columns
train_df.head(5)


def text_to_words(text):
    letters_only = re.sub("[^a-zA-Z]"," ", text)
    lower_case = letters_only.lower()
    words = lower_case.split()
    large_words = [w for w in words if len(w) > 3 if len(w) < 10]
    return (" ".join(large_words))

train_df["site"] = train_df["site"].fillna("NA")

num_alerts = train_df['serialNum'].shape[0]
clean_train_alerts = []
for i in range(0, num_alerts):
    clean_train_alerts.append(train_df['serviceType'][i] + " " \
        + text_to_words(train_df['symptom'][i]) + " " + train_df['site'][i] + " " \
        + train_df['DayofWeek '][i] + " " + text_to_words(train_df['summary'][i]))

print('Creating bag of words...\n')
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(analyzer = "word", tokenizer = None, \
                             preprocessor = None, \
                             stop_words = None, \
                             max_features = 5000)
train_data_features = vectorizer.fit_transform(clean_train_alerts)

train_data_features = train_data_features.toarray()

print(train_data_features.shape)

vocab = vectorizer.get_feature_names()
print(vocab)

print("Training the Random forest...")
from sklearn.ensemble import RandomForestClassifier

# initialize the random forest classifier with number of trees
forest = RandomForestClassifier(n_estimators = 20)

forest = forest.fit(train_data_features, train_df['Actionability'])

test = pd.read_csv('./input/Test.csv', delimiter=',')

test["site"] = test["site"].fillna("NA")

test.shape
num_alerts = len(test['symptom'])
test["symptom"] = test["symptom"].fillna("NA")
clean_test_alerts = []
for i in range(0, num_alerts):
    clean_test_alerts.append(test['serviceType'][i] + " " \
        + text_to_words(test['symptom'][i]) + " " \
        + test['site'][i] + " " + test['DayofWeek '][i] + " " + text_to_words(test['symptom'][i]))


# Get a bag of words for the test set, and convert to a numpy array
test_data_features = vectorizer.transform(clean_test_alerts)
test_data_features = test_data_features.toarray()

result = forest.predict(test_data_features)


output = pd.DataFrame(data={"serialNum":test["serialNum"], "Actionability":result})
output.loc[output["Actionability"] == "Non-actionable", "Actionability"] = 0
output.loc[output["Actionability"] == "Actionable", "Actionability"] = 1
out = output[['serialNum', 'Actionability']]

out.to_csv("Alerts_Actionability.csv", index=False)

print("Time elapsed: %.2f sec" % (time.time() - t0))