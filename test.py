# Unit Tests
# Please write the unit tests to check whether your model:
# * is reproducible
# * can handle missing values
# * can handle new category levels at prediction time
# * returns results in the expected format
# * other useful unit tests you may think of (if time allows)
# from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from custom_log_regression import CustomLogRegression

DATA_PATH = './data/DR_Demo_Lending_Club_reduced.csv'
RND_STATE = 42


test_size = 0.3

clf = CustomLogRegression(random_state=RND_STATE)
X_train, X_test, y_train, y_test = clf.read_data(data_path=DATA_PATH, y_col='is_bad', index_col='Id')
params = clf.tune_parameters(X_train, y_train)
print(params)

clf.fit(X_train, y_train)
metrics = clf.evaluate(X_test, y_test)

print('Accuracy {:.2}, F1 Score {:.2}, Logistic Loss {:.2}'.format(
    *metrics.values()
))
