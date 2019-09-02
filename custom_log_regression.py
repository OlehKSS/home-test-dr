"""Prediction class implementation based on logistic regression."""

from category_encoders.one_hot import OneHotEncoder
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class CustomLogRegression:
    """
    Logistic regression with additional methods.

    Parameters
    ----------
    random_state : int, float
        Random state to make classifier reproducible. Default is None.
    n_jobs : int
        Number of processes to use for the classifier training.    
    """
    def __init__(self, random_state=None, n_jobs=1):
        self._random_state = random_state
        self._n_jobs = n_jobs
        self._clf_params = None
        self._clf = None
    
    def read_data(self, data_path, y_col, index_col=None, skip_cols=None,
        test_size=0.3):
        """
        Read a csv file with data.

        Categorical variables will be encoded according to the one hot
        scheme.

        Parameters
        ----------
        data_path : str
            Path to the csv file with data.
        y_col : str
            Ground truth labels with values of 0 and 1.
        index_col : int, sequence or bool, optional
            Column to use as the row labels of the DataFrame. If a sequence is given,
            MultiIndex is used.
        skip_cols : list
            List of features / columns to exclude from the data.
        test_size : float
            Should be between 0.0 and 1.0 and represent the proportion of the dataset
            to include in the test split.
        """
        data = pd.read_csv(data_path, index_col=index_col, na_values=['NONE', 'na'])
        # drop columns with where N/As constitute around 10% of all entries
        na_max_percent = 0.1
        nas = data.isna().sum()
        excessive_na_cols = set(nas[nas > na_max_percent * len(data)].index).union(
            set(skip_cols)
        )
        data_cols = set(data.columns).difference(excessive_na_cols)

        if y_col not in data_cols:
            raise ValueError(f'Too many enties without the labels {y_col}')

        numeric_cols = set(data._get_numeric_data().columns).difference(
            excessive_na_cols
        )
        # since y_col contains 0, 1 it should be numeric
        categorical_cols = data_cols - numeric_cols
        numeric_cols.remove(y_col)

        data = data.loc[:, data_cols]
        data = data.dropna()
        X = data.loc[:, numeric_cols.union(categorical_cols)]
        y = data.is_bad.values

        # encode categorical variables
        encoder = OneHotEncoder(cols=categorical_cols, use_cat_names=True)
        X = encoder.fit_transform(X)
        data_splits = train_test_split(X, y, test_size=test_size,
            random_state=self._random_state)

        return data_splits

    def fit(self, X, y):
        """
        Fit on training data.

        Parameters
        ----------
        X : pandas.DataFrame
            Input features.
        y : numpy.array
            Ground truth labels as a numpy array of 0s and 1s.
        """
        scaler = StandardScaler()
        scaler.fit(X)
        X_scaled = scaler.transform(X)

        if self._clf_params is not None:
            C = self._clf_params['C']
        else:
            C = 1.0

        self._clf = LogisticRegression(random_state=self._random_state, C=C,
            solver='lbfgs', class_weight='balanced', n_jobs=self._n_jobs)

        self._clf.fit(X_scaled, y)
    
    def predict(self, X):
        """
        Predict class labels on new data.

        Parameters
        ----------
        X : pandas.DataFrame
            Input features.
        
        Returns
        -------
        out : numpy.ndarray
        """
        return self._clf.predict(X)

    def predict_proba(self, X):
        """
        Predict the probability of each label.

        Parameters
        ----------
        X : pandas.DataFrame
            Input features.
        
        Returns
        -------
        proba : numpy.ndarray
        """
        return self._clf.predict_proba(X)

    def evaluate(self, X, y):
        """
        Get the following metrics: F1-score, LogLoss.

        Parameters
        ----------
        X : pandas.DataFrame
            Input features.
        y : numpy.array
            Ground truth labels as a numpy array of 0s and 1s.

        Returns
        -------
        metrics : dict
            {'accuracy': 0.8, 'f1_score': 0.3, 'logloss': 0.7}
        """
        metrics = {}
        scaler = StandardScaler()
        scaler.fit(X)
        X_scaled = scaler.transform(X)

        y_pred = self._clf.predict(X_scaled)
        metrics['accuracy'] = accuracy_score(y, y_pred)
        metrics['f1_score'] = f1_score(y, y_pred)
        metrics['logloss'] = log_loss(y, y_pred)

        return metrics
    
    def tune_parameters(self, X, y, n_folds=5):
        """
        Run K-fold cross validation to choose the best parameters.

        Parameters
        ----------
        X : pandas.DataFrame
            Input features.
        y : numpy.array
            Ground truth labels as a numpy array of 0s and 1s.
        n_folds : int
            The number of folds in K-fold cross-validation. Default is 5.

        Returns
        -------
        params : dict
            {'tol': 0.02, 'fit_intercept': False, 'solver': 'sag', 'scores':
                {'f1_score': 0.3, 'logloss': 0.7}}
        """
        regularization = np.logspace(-7, 2, 100)
        accuracy_scores = np.zeros_like(regularization)
        f1_scores = np.zeros_like(regularization)
        log_loss_scores = np.zeros_like(regularization)

        for index, c in enumerate(regularization):
            clf = LogisticRegression(random_state=self._random_state, C=c,
                solver='lbfgs', n_jobs=self._n_jobs, class_weight='balanced')
            estimator = make_pipeline(StandardScaler(), clf)
            _accuracy_scores = cross_val_score(estimator, X, y, cv=n_folds)
            _f1_scs = cross_val_score(estimator, X, y, cv=n_folds, scoring='f1',
                n_jobs=self._n_jobs)
            _log_loss_scs = cross_val_score(estimator, X, y, cv=n_folds, scoring='neg_log_loss',
                n_jobs=self._n_jobs)
            accuracy_scores[index] = _accuracy_scores.mean()
            f1_scores[index] = _f1_scs.mean()
            log_loss_scores[index] = -1 * _log_loss_scs.mean()

        scores = {'accuracy': accuracy_scores, 'f1_score': f1_scores, 'logloss': log_loss_scores}
        params = {'C': regularization[np.argmax(f1_scores)], 'scores': scores}
        self._clf_params = {'C': regularization[np.argmax(f1_scores)]}

        return params
            
