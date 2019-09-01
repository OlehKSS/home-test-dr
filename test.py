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

data = pd.read_csv(DATA_PATH)
print(f'Initial data size {data.shape}')
# data = data.dropna()

# drop Id, is_bad
numeric_cols = ['annual_inc', 'collections_12_mths_ex_med', 'debt_to_income', 'debt_to_income',
    'delinq_2yrs', 'revol_util', 'emp_length', 'total_acc', 'inq_last_6mths', 'mths_since_last_delinq',
    'mths_since_last_major_derog', 'mths_since_last_record', 'open_acc', 'pub_rec', 'revol_bal']

categorical_cols = ['addr_state', 'home_ownership', 'initial_list_status', 'zip_code', 'pymnt_plan',
    'policy_code', 'verification_status', 'purpose_cat']

# clean non-numeric values in numeric columns
# not correctly placed
data = data[:, numeric_cols]
data = data.loc[data.loc[:, 'emp_length'].str.isnumeric().values]
data = data.dropna()

print(f'Data size {data.shape}')

X = data.loc[:, numeric_cols]
y = data.is_bad.values

# 'na' values should be deleted

test_size = 0.3

X_train, X_test, y_train, y_test = train_test_split(X, y,
    test_size=test_size, random_state=RND_STATE)

clf = CustomLogRegression(random_state=RND_STATE)
clf.fit(X_train, y_train)
metrics = clf.evaluate(X_test, y_test)

print('Accuracy {:.2}, F1 Score {:.2}, Logistic Loss {:.2}'.format(
    *metrics.values()
))

params = clf.tune_parameters(X_train, y_train)
print(params)
# test, train split
