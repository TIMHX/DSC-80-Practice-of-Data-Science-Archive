# lab.py


import os
import pandas as pd
import numpy as np
import util


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def first_round():
    """
    :return: list with two values
    >>> out = first_round()
    >>> isinstance(out, list)
    True
    >>> out[0] < 1
    True
    >>> out[1] == "NR" or out[1] == "R"
    True
    """
    return [0.897, 'NR']
    ...


def second_round():
    """
    :return: list with three values
    >>> out = second_round()
    >>> isinstance(out, list)
    True
    >>> out[0] < 1
    True
    >>> out[1] == "NR" or out[1] == "R"
    True
    >>> out[2] == "ND" or out[2] == "D"
    True
    """
    return [0.021, 'R', 'D']
    ...


# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def verify_child(heights):
    """
    Returns a series of p-values assessing the missingness
    of child-height columns on father height.

    >>> fp = os.path.join('data', 'missing_heights.csv')
    >>> heights = pd.read_csv(fp)
    >>> out = verify_child(heights)
    >>> out['child_50'] < out['child_95']
    True
    >>> out['child_5'] > out['child_50']
    True
    """
    heights_c = heights.copy()
    out = []
    names = [
        'child_95', 'child_90', 'child_75', 'child_50', 'child_25', 'child_10',
        'child_5'
    ]
    for i in heights_c[names]:
        distr, obs = util.permutation_test(
            heights_c.assign(is_null=heights_c[i].isnull()), 'father',
            'is_null', util.ks, N=100)
        pval = (distr >= obs).mean()
        out.append(pval)
    return pd.Series(out, index = names)
    ...


def missing_data_amounts():
    """
    Returns a list of multiple choice answers.
    :Example:
    >>> set(missing_data_amounts()) <= set(range(1,6))
    True
    """
    return [1, 2, 5]
    ...


# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


def cond_single_imputation(new_heights):
    """
    cond_single_imputation takes in a dataframe with columns 
    father and child (with missing values in child) and imputes 
    single-valued mean imputation of the child column, 
    conditional on father. Your function should return a Series.

    :Example:
    >>> fp = os.path.join('data', 'missing_heights.csv')
    >>> df = pd.read_csv(fp)
    >>> df['child'] = df['child_50']
    >>> out = cond_single_imputation(df)
    >>> out.isnull().sum() == 0
    True
    >>> (df.child.std() - out.std()) > 0.5
    True
    """
    df_c = new_heights.copy()
    df_c['father_q'] = pd.qcut(df_c.father, q=5)
    return df_c.groupby('father_q')['child'].transform(lambda x: x.fillna(x.mean()))
    ...


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


def quantitative_distribution(child, N):
    """
    quantitative_distribution that takes in a Series and an integer 
    N > 0, and returns an array of N samples from the distribution of 
    values of the Series as described in the question.
    :Example:
    >>> fp = os.path.join('data', 'missing_heights.csv')
    >>> df = pd.read_csv(fp)
    >>> child = df['child_50']
    >>> out = quantitative_distribution(child, 100)
    >>> out.min() >= 56
    True
    >>> out.max() <= 79
    True
    >>> np.isclose(out.mean(), child.mean(), atol=1)
    True
    >>> np.isclose(out.std(), 3.5, atol=0.65)
    True
    """
    child_c = child.copy().dropna().reset_index().drop('index', axis=1)
    hist, edges = np.histogram(child_c, bins=10)
    distribution = hist / sum(hist)
    mapping = []
    out = []
    for i in range(1, len(edges)):
        mapping.append((edges[i - 1], edges[i]))
    dice = np.random.choice(len(mapping), N, p=distribution)
    for i in dice:
        out.append(np.random.uniform(mapping[i][0], mapping[i][1]))
    return pd.Series(out)
    ...


def impute_height_quant(child):
    """
    impute_height_quant takes in a Series of child heights 
    with missing values and imputes them using the scheme in
    the question.
    :Example:
    >>> fp = os.path.join('data', 'missing_heights.csv')
    >>> df = pd.read_csv(fp)
    >>> child = df['child_50']
    >>> out = impute_height_quant(child)
    >>> out.isnull().sum() == 0
    True
    >>> np.isclose(out.mean(), child.mean(), atol=0.5)
    True
    >>> np.isclose(out.mean(), child.mean(), atol=0.2)
    True
    >>> np.isclose(out.std(), child.std(), atol=0.15)
    True
    """
    col = child.copy()
    num_null = col.isnull().sum()
    fill_values = quantitative_distribution(col, num_null)
    fill_values.index = col.loc[col.isnull()].index
    return col.fillna(fill_values.to_dict())
    ...


# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------


def answers():
    """
    Returns two lists with your answers
    :return: Two lists: one with your answers to multiple choice questions
    and the second list has 6 websites that satisfy given requirements.
    >>> mc_answers, websites = answers()
    >>> len(mc_answers)
    4
    >>> len(websites)
    6
    """
    return [1, 2, 2, 1], ['https://en.wikipedia.org/', 'https://www.sina.com.cn/', 'diply.com', 'facebook.com',
                          'https://vk.com/', 'https://www.tmall.com/']
    ...
