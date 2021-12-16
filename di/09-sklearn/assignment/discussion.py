# discussion.py


import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------


from sklearn.base import BaseEstimator, TransformerMixin

class LowStdColumnDropper(BaseEstimator, TransformerMixin):

    def __init__(self, thresh=0):
        '''
        Drops columns whose standard deviation is less than thresh.
        '''
        self.thresh = thresh

    def fit(self, X, y=None):
        if isinstance(X, np.ndarray):
            self.columns_ =list(X.std(axis=0) >= self.thresh)
        else:
            self.columns_ = list(X.std(axis=0) >= self.thresh)
        return self
    def transform(self,  X, y=None):
        if isinstance(X, np.ndarray):
            out = X[:, self.columns_]
        else:
            out  = X.iloc[:, self.columns_]
        return out
        ...
