# lab.py


import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.preprocessing import Binarizer, QuantileTransformer, FunctionTransformer


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def best_transformation():
    """
    Returns an integer corresponding to the correct option.

    :Example:
    >>> best_transformation() in [1,2,3,4]
    True
    """
    return 1
    ...


# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def create_ordinal(df):
    """
    create_ordinal takes in diamonds and returns a dataframe of ordinal
    features with names ordinal_<col> where <col> is the original
    categorical column name.

    :Example:
    >>> diamonds = sns.load_dataset('diamonds')
    >>> out = create_ordinal(diamonds)
    >>> set(out.columns) == {'ordinal_cut', 'ordinal_clarity', 'ordinal_color'}
    True
    >>> np.unique(out['ordinal_cut']).tolist() == [0, 1, 2, 3, 4]
    True
    """
    df_c = df.copy()
    out = pd.DataFrame()
    mapper_cut = {'Fair': 0, 'Good': 1, 'Very Good': 2, 'Premium': 3, 'Ideal': 4}
    mapper_color = {'J': 0, 'I': 1, 'H': 2, 'G': 3, 'F': 4, 'E': 5, 'D': 6}
    mapper_clarity = {'I1': 0, 'SI2': 1, 'SI1': 2, 'VS2': 3, 'VS1': 4, 'VVS2': 5, 'VVS1': 6, 'IF': 7}
    out['ordinal_cut'] = df_c['cut'].replace(mapper_cut)
    out['ordinal_color'] = df_c['color'].replace(mapper_color)
    out['ordinal_clarity'] = df_c['clarity'].replace(mapper_clarity)
    return out
    ...


# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


def create_one_hot(df):
    """
    create_one_hot takes in diamonds and returns a dataframe of one-hot 
    encoded features with names one_hot_<col>_<val> where <col> is the 
    original categorical column name, and <val> is the value found in 
    the categorical column <col>.

    :Example:
    >>> diamonds = sns.load_dataset('diamonds')
    >>> out = create_one_hot(diamonds)
    >>> out.shape == (53940, 20)
    True
    >>> out.columns.str.startswith('one_hot').all()
    True
    >>> out.isin([0,1]).all().all()
    True
    """

    def helper(col):
        labels_dense = col.copy()
        uni_num = len(col.unique())
        index_offset = np.arange(labels_dense.shape[0]) * uni_num
        labels_one_hot = np.zeros((len(col), uni_num))
        labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
        return labels_one_hot

    ordinal = create_ordinal(df)
    out_cut = helper(ordinal['ordinal_cut'])
    out_color = helper(ordinal['ordinal_color'])
    out_clarity = helper(ordinal['ordinal_clarity'])
    df_cut = pd.DataFrame(out_cut)
    df_color = pd.DataFrame(out_color)
    df_clarity = pd.DataFrame(out_clarity)

    df_cut.columns = ['one_hot_cut_Fair', 'one_hot_cut_Good', 'one_hot_cut_Very Good', 'one_hot_cut_Premium',
                      'one_hot_cut_Ideal']
    df_color.columns = ['one_hot_color_J', 'one_hot_color_I', 'one_hot_color_H', 'one_hot_color_G', 'one_hot_color_F',
                        'one_hot_color_E', 'one_hot_color_D']
    df_clarity.columns = ['one_hot_clarity_I1', 'one_hot_clarity_SI2', 'one_hot_clarity_SI1', 'one_hot_clarity_VS2', 'one_hot_clarity_VS1',
                          'one_hot_clarity_VVS2', 'one_hot_clarity_VVS1', 'one_hot_clarity_IF']
    return pd.concat([df_cut, df_color, df_clarity], axis=1)
    ...


def create_proportions(df):
    """
    create_proportions takes in diamonds and returns a 
    dataframe of proportion-encoded features with names 
    proportion_<col> where <col> is the original 
    categorical column name.

    >>> diamonds = sns.load_dataset('diamonds')
    >>> out = create_proportions(diamonds)
    >>> out.shape[1] == 3
    True
    >>> out.columns.str.startswith('proportion_').all()
    True
    >>> ((out >= 0) & (out <= 1)).all().all()
    True
    """
    ordinal = create_ordinal(df)
    cut_tab = pd.crosstab(ordinal.index, ordinal.ordinal_cut).sum(axis=0) / len(ordinal)
    color_tab = pd.crosstab(ordinal.index, ordinal.ordinal_color).sum(axis=0) / len(ordinal)
    clarity_tab = pd.crosstab(ordinal.index, ordinal.ordinal_clarity).sum(axis=0) / len(ordinal)
    out = pd.DataFrame()
    out['proportion_cut'] = ordinal.ordinal_cut.apply(lambda x: cut_tab[x])
    out['proportion_color'] = ordinal.ordinal_color.apply(lambda x: color_tab[x])
    out['proportion_clarity'] = ordinal.ordinal_clarity.apply(lambda x: clarity_tab[x])
    return out
    ...


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


def create_quadratics(df):
    """
    create_quadratics that takes in diamonds and returns a dataframe 
    of quadratic-encoded features <col1> * <col2> where <col1> and <col2> 
    are the original quantitative columns 
    (col1 and col2 should be distinct columns).

    :Example:
    >>> diamonds = sns.load_dataset('diamonds')
    >>> out = create_quadratics(diamonds)
    >>> out.columns.str.contains(' * ').all()
    True
    >>> ('x * z' in out.columns) or ('z * x' in out.columns)
    True
    >>> out.shape[1] == 15
    True
    """
    df_c = df.copy()[['carat', 'depth', 'table', 'x', 'y', 'z']]
    out = pd.DataFrame()
    for i in range(len(df_c.columns)):
        for j in range(i + 1, len(df_c.columns)):
            col_1 = df_c.columns[i]
            col_2 = df_c.columns[j]
            out['{} * {}'.format(
                df_c.columns[i],
                df_c.columns[j])] = df_c.loc[:, col_1] * df_c.loc[:, col_2]
    return out
    ...


# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------


def comparing_performance():
    """
    Hard coded answers to comparing_performance.
    :Example:
    >>> out = comparing_performance()
    >>> len(out) == 6
    True
    >>> import numbers
    >>> isinstance(out[0], numbers.Real)
    True
    >>> all(isinstance(x, str) for x in out[2:-1])
    True
    >>> 0 <= out[-1] <= 1
    True
    """

    # create a model per variable => (variable, R^2, RMSE) table
    return [0.8493305264354858, 1548.5331930613174, 'x', 'carat * x',
            'color', 0.041]
    ...


# ---------------------------------------------------------------------
# QUESTION 6
# ---------------------------------------------------------------------


class TransformDiamonds(object):

    def __init__(self, diamonds):
        self.data = diamonds

    def transformCarat(self, data):
        """
        transformCarat takes in a dataframe like diamonds 
        and returns a binarized carat column (an np.ndarray).
        :Example:
        >>> diamonds = sns.load_dataset('diamonds')
        >>> out = TransformDiamonds(diamonds)
        >>> transformed = out.transformCarat(diamonds)
        >>> isinstance(transformed, np.ndarray)
        True
        >>> transformed[172, 0] == 1
        True
        >>> transformed[0, 0] == 0
        True
        """
        bi = Binarizer(threshold=1-0.0000001)
        binarized = bi.transform(data[['carat']])
        return binarized
        ...

    def transform_to_quantile(self, data):
        """
        transform_to_quantiles takes in a dataframe like diamonds 
        and returns an np.ndarray of quantiles of the weight 
        (i.e. carats) of each diamond.
        :Example:
        >>> diamonds = sns.load_dataset('diamonds')
        >>> out = TransformDiamonds(diamonds.head(10))
        >>> transformed = out.transform_to_quantile(diamonds)
        >>> isinstance(transformed, np.ndarray)
        True
        >>> 0.2 <= transformed[0,0] <= 0.5
        True
        >>> np.isclose(transformed[1,0], 0, atol=1e-06)
        True
        """
        qt = QuantileTransformer()
        qt.fit(self.data[['carat']])
        return qt.transform(data[['carat']])
        ...

    def transform_to_depth_pct(self, data):
        """
        transform_to_volume takes in a dataframe like diamonds 
        and returns an np.ndarray consisting of the approximate 
        depth percentage of each diamond.
        :Example:
        >>> diamonds = sns.load_dataset('diamonds').drop(columns='depth')
        >>> out = TransformDiamonds(diamonds)
        >>> transformed = out.transform_to_depth_pct(diamonds)
        >>> len(transformed.shape) == 1
        True
        >>> np.isclose(transformed[0], 61.286, atol=0.0001)
        True
        """
        X = np.array(data.copy()[['x', 'y', 'z']])

        def func(X):
            return X[:, 2] / ((X[:, 0] + X[:, 1]) / 2) * 100

        transformer = FunctionTransformer(func)
        transformer.fit(self.data)
        return transformer.transform(X)
        ...
