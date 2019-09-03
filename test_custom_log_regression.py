# Unit Tests
from math import isclose

import numpy as np

from custom_log_regression import CustomLogRegression


DATA_PATH = './data/DR_Demo_Lending_Club_reduced.csv'
RND_STATE = 42
N_JOBS = 2
TEST_SIZE = 0.3
Y_COL = 'is_bad'
INDEX_COL = 'Id'
SKIP_COLS = ('zip_code', 'addr_state')
METRICS = ('accuracy', 'f1_score', 'logloss')


def test_class_instantiation_defaults():
    clf = CustomLogRegression()

    assert clf._random_state is None
    assert clf._n_jobs == 1


def test_class_instantiation():
    clf = CustomLogRegression(random_state=RND_STATE,
                              n_jobs=N_JOBS)

    assert clf._random_state == RND_STATE
    assert clf._n_jobs == N_JOBS


def test_read_data():
    clf = CustomLogRegression(random_state=RND_STATE,
                              n_jobs=N_JOBS)
    data_parts = clf.read_data(data_path=DATA_PATH, y_col=Y_COL,
                               index_col=INDEX_COL, skip_cols=SKIP_COLS,
                               test_size=TEST_SIZE)

    # should split data in 4 parts
    assert len(data_parts) == 4

    X_train, X_test, y_train, y_test = data_parts

    # 30% of the samples should be in X_test
    assert isclose(len(X_train) / 0.7, len(X_test) / 0.3)

    # should skip specified features
    for c in SKIP_COLS:
        assert c not in X_train.columns
        assert c not in X_test.columns

    # should not contain N/A values
    assert X_train.isna().sum().sum() == 0
    assert X_test.isna().sum().sum() == 0
    assert np.isnan(y_train).sum() == 0
    assert np.isnan(y_test).sum() == 0


def test_tune_parameters():
    C = 0.0008
    clf = CustomLogRegression(random_state=RND_STATE,
                              n_jobs=N_JOBS)
    X_train, X_test, y_train, y_test = clf.read_data(
        data_path=DATA_PATH, y_col=Y_COL, index_col=INDEX_COL,
        skip_cols=SKIP_COLS, test_size=TEST_SIZE)
    params = clf.tune_parameters(X_train, y_train)

    assert type(params) is dict
    assert isclose(params['C'], C, rel_tol=0.1)
    assert type(params['scores']) is dict
    assert tuple(params['scores'].keys()) == METRICS


def test_fit_and_evaluate():
    EXPECTED_F1_SCORE = 0.31
    clf = CustomLogRegression(random_state=RND_STATE,
                              n_jobs=N_JOBS)
    X_train, X_test, y_train, y_test = clf.read_data(
        data_path=DATA_PATH, y_col=Y_COL, index_col=INDEX_COL,
        skip_cols=SKIP_COLS, test_size=TEST_SIZE)
    clf.tune_parameters(X_train, y_train)
    clf.fit(X_train, y_train)
    metrics = clf.evaluate(X_test, y_test)

    assert type(metrics) is dict
    assert tuple(metrics.keys()) == METRICS
    assert isclose(metrics['f1_score'], EXPECTED_F1_SCORE, rel_tol=0.1)
