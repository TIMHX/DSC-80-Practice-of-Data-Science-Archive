# lab.py


import pandas as pd
import numpy as np
import seaborn as sns
import os

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV

# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def simple_pipeline(data):
    '''
    simple_pipeline takes in a dataframe like data and returns a tuple
    consisting of the pipeline and the predictions your model makes
    on data (as trained on data).
    :Example:
    >>> fp = os.path.join('data', 'toy.csv')
    >>> data = pd.read_csv(fp)
    >>> pl, preds = simple_pipeline(data)
    >>> isinstance(pl, Pipeline)
    True
    >>> isinstance(pl.steps[-1][1], LinearRegression)
    True
    >>> isinstance(pl.steps[0][1], FunctionTransformer)
    True
    >>> preds.shape[0] == data.shape[0]
    True
    '''
    c2 = np.array(data.c2)
    y =  np.array(data.y)
    pl = Pipeline(steps=[
        ('scaler', FunctionTransformer(np.log)),
        ('lin-reg', LinearRegression())
    ])
    pl.fit(c2.reshape(-1, 1), y)
    predict = pl.predict(c2.reshape(-1, 1))
    return (pl, predict)
    ...


# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def multi_type_pipeline(data):
    '''
    multi_type_pipeline that takes in a dataframe like data and
    returns a tuple consisting of the pipeline and the predictions
    your model makes on data (as trained on data).
    :Example:
    >>> fp = os.path.join('data', 'toy.csv')
    >>> data = pd.read_csv(fp)
    >>> pl, preds = multi_type_pipeline(data)
    >>> isinstance(pl, Pipeline)
    True
    >>> isinstance(pl.steps[-1][1], LinearRegression)
    True
    >>> isinstance(pl.steps[0][1], ColumnTransformer)
    True
    >>> data.shape[0] == preds.shape[0]
    True
    '''
    num_feat_1 = ['c1']
    num_transformer_1 = Pipeline(steps=[
        ('scaler', None),
    ])

    num_feat_2 = ['c2']
    num_transformer_2 = Pipeline(steps=[
        ('scaler', FunctionTransformer(np.log)),
    ])

    cat_feat = ['group']
    cat_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder())  # output from Ordinal becomes input to OneHot
    ])

    preproc = ColumnTransformer(
        transformers=[
            ('num_1', num_transformer_1, num_feat_1),
            ('num_2', num_transformer_2, num_feat_2),
            ('cat', cat_transformer, cat_feat)
        ])

    pl = Pipeline(steps=[('preprocessor', preproc), ('regressor', LinearRegression())])

    pl.fit(data[['group', 'c1', 'c2']], data.y)
    predict = pl.predict(data[['group', 'c1', 'c2']])
    return (pl, predict)
    ...


# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


from sklearn.base import BaseEstimator, TransformerMixin


class StdScalerByGroup(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        """
        :Example:
        >>> cols = {'g': ['A', 'A', 'B', 'B'], 'c1': [1, 2, 2, 2], 'c2': [3, 1, 2, 0]}
        >>> X = pd.DataFrame(cols)
        >>> std = StdScalerByGroup().fit(X)
        >>> std.grps_ is not None
        True
        """
        # X may not be a pandas dataframe (e.g. a np.array)
        df = pd.DataFrame(X)

        # A dictionary of means/standard-deviations for each column, for each group.
        try:
            mean_df = df.groupby(df.columns[0]).mean()
            sd_df = df.groupby(df.columns[0]).std()
            combine_df = mean_df.merge(sd_df, on= df.columns[0])
            combine_df.columns = ['c1_mean', 'c2_mean', 'c1_sd', 'c2_sd']
            self.grps_ = dict(combine_df)
        except:
            self.grps_ = pd.DataFrame()

        return self

    def transform(self, X, y=None):
        """
        :Example:
        >>> cols = {'g': ['A', 'A', 'B', 'B'], 'c1': [1, 2, 3, 4], 'c2': [1, 2, 3, 4]}
        >>> X = pd.DataFrame(cols)
        >>> std = StdScalerByGroup().fit(X)
        >>> out = std.transform(X)
        >>> out.shape == (4, 2)
        True
        >>> np.isclose(out.abs(), 0.707107, atol=0.001).all().all()
        True
        """

        try:
            getattr(self, "grps_")
        except AttributeError:
            raise RuntimeError("You must fit the transformer before tranforming the data!")

        # Define a helper function here?
        zscore = lambda x: (x - x.mean())/x.std()

        # X may not be a dataframe (e.g. np.array)
        df = pd.DataFrame(X)

        return df.groupby(df.columns[0]).transform(zscore)


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


def eval_toy_model():
    """
    hardcoded answers to question 4
    :Example:
    >>> out = eval_toy_model()
    >>> len(out) == 3
    True
    """
    return [tuple([2.755108697451812, 0.39558507345910754]),
            tuple([2.3148336164355277, 0.5733249315673331]),
            tuple([2.3157339477882385, 0.5729929650348397])]
    ...


# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------
def model_helper(galton, mod):
    X = galton.drop('childHeight', axis=1)
    y = galton.childHeight
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state= 42)

    rmse_train_list = []
    rmse_test_list = []
    for i in range(1, 21):
        if mod == 'D':
            clf = DecisionTreeRegressor(max_depth=i)
        elif mod == 'K':
            clf = KNeighborsRegressor(n_neighbors=i)
        clf.fit(X_train, y_train)

        pred_train = clf.predict(X_train)
        pred_test = clf.predict(X_test)
        rmse_train = (np.mean((pred_train - y_train) ** 2))**0.5
        rmse_test = (np.mean((pred_test - y_test) ** 2))**0.5
        rmse_train_list.append(rmse_train)
        rmse_test_list.append(rmse_test)

    data_tuples = list(zip(rmse_train_list, rmse_test_list))
    out = pd.DataFrame(data_tuples, columns=['train_err', 'test_err'])
    out.index = np.arange(1, len(out) + 1)
    return out

def tree_reg_perf(galton):
    """

    :Example:
    >>> galton_fp = os.path.join('data', 'galton.csv')
    >>> galton = pd.read_csv(galton_fp)
    >>> out = tree_reg_perf(galton)
    >>> out.columns.tolist() == ['train_err', 'test_err']
    True
    >>> out['train_err'].iloc[-1] < out['test_err'].iloc[-1]
    True
    """
    out = model_helper(galton, 'D')
    return out
    ...


def knn_reg_perf(galton):
    """
    :Example:
    >>> galton_fp = os.path.join('data', 'galton.csv')
    >>> galton = pd.read_csv(galton_fp)
    >>> out = knn_reg_perf(galton)
    >>> out.columns.tolist() == ['train_err', 'test_err']
    True
    """
    out = model_helper(galton, 'K')
    return out
    ...


# ---------------------------------------------------------------------
# QUESTION 6
# ---------------------------------------------------------------------


def titanic_model(tit):
    """
    :Example:
    >>> fp = os.path.join('data', 'titanic.csv')
    >>> data = pd.read_csv(fp)
    >>> pl = titanic_model(data)
    >>> isinstance(pl, Pipeline)
    True
    >>> from sklearn.base import BaseEstimator
    >>> isinstance(pl.steps[-1][-1], BaseEstimator)
    True
    >>> preds = pl.predict(data.drop('Survived', axis=1))
    >>> ((preds == 0)|(preds == 1)).all()
    True
    """
    titanic = tit.copy()

    titanic = tit.copy()
    titanic['Fare'].fillna(titanic['Fare'].mean(), inplace=True)
    titanic['Name'] = titanic['Name'].apply(lambda x: x.split(' ')[0])

    def deal_name(X):
        return X

    X = titanic.drop('Survived', axis=1)
    y = titanic.Survived

    num_feat_1 = ['Age']
    num_transformer_1 = Pipeline(steps=[
        ('scaler', StdScalerByGroup()),
        #         ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
    ])

    num_feat_2 = ['Siblings/Spouses Aboard', 'Parents/Children Aboard', 'Fare']
    num_transformer_2 = Pipeline(steps=[
        ('imp', SimpleImputer(strategy='constant', fill_value=0)),
        #         ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False)),
    ])

    cat_feat = ['Pclass', 'Sex']
    cat_transformer = Pipeline(steps=[
        ('imp', SimpleImputer(strategy='constant', fill_value='NULL')),
        #         ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False)),
        #                 ('pca', PCA(svd_solver='full', n_components=0.99))
    ])

    cat_feat_2 = ['Name']
    cat_transformer_2 = Pipeline(steps=[
        ('imp', SimpleImputer(strategy='constant', fill_value='NULL')
         ), ('clean', FunctionTransformer(deal_name, validate=False))
        #         ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False)),
        #                 ('pca', PCA(svd_solver='full', n_components=0.99))
    ])

    preproc = ColumnTransformer(transformers=[(
        'num_1', num_transformer_1,
        num_feat_1), ('num_2', num_transformer_2,
                      num_feat_2), (
        'cat', cat_transformer,
        cat_feat), ('cat_2', cat_transformer_2, cat_feat_2)])

    pl = Pipeline(steps=[
        ('preprocessor', preproc),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False)),
        #                                  ('pca', PCA(svd_solver='full', n_components=0.99)),
        ('regressor', DecisionTreeClassifier())
    ])
    pl.fit(X, y)
    return pl
    ...


# ---------------------------------------------------------------------
# QUESTION 7
# ---------------------------------------------------------------------


def json_reader(file, iterations):
    """
    :Example
    >>> fp = os.path.join('data', 'reviews.json')
    >>> reviews, labels = json_reader(fp, 5000)
    >>> isinstance(reviews, list)
    True
    >>> isinstance(labels, list)
    True
    >>> len(labels) == len(reviews)
    True
    """
    ...


def create_classifier_multi(X, y):
    """
    :Example
    >>> fp = os.path.join('data', 'reviews.json')
    >>> reviews, labels = json_reader(fp, 5000)
    >>> trial = create_classifier_multi(reviews, labels)
    >>> isinstance(trial, Pipeline)
    True

    """
    ...


def to_binary(labels):
    """
    :Example
    >>> lst = [1, 2, 3, 4, 5]
    >>> to_binary(lst)
    >>> print(lst)
    [0, 0, 0, 1, 1]
    """
    ...


def create_classifier_binary(X, y):
    """
    :Example
    >>> fp = os.path.join('data', 'reviews.json')
    >>> reviews, labels = json_reader(fp, 5000)
    >>> to_binary(labels)
    >>> trial = create_classifier_multi(reviews, labels)
    >>> isinstance(trial, Pipeline)
    True

    """
    ...
