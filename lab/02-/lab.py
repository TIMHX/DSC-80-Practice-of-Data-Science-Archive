# lab.py


import pandas as pd
import numpy as np
import os


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def data_load(scores_fp):
    """
    follows different steps to create a dataframe
    :param scores_fp: file name as a string
    :return: a dataframe
    >>> scores_fp = os.path.join('data', 'scores.csv')
    >>> scores = data_load(scores_fp)
    >>> isinstance(scores, pd.DataFrame)
    True
    >>> list(scores.columns)
    ['attempts', 'highest_score']
    >>> isinstance(scores.index[0], int)
    False
    """
    fields = ['name', 'tries', 'highest_score','sex']
    score = pd.read_csv(scores_fp,  usecols=fields)
    score = score.drop(columns = ['sex'])
    score = score.rename(columns = {'name': 'firstname', 'tries': 'attempts'})
    score = score.rename(index=score['firstname']).drop(columns = 'firstname')
    return score
    ...


def pass_fail(scores):
    """
    modifies the scores dataframe by adding one more column satisfying
    conditions from the write up.
    :param scores: dataframe from the question above
    :return: dataframe with additional column pass
    >>> scores_fp = os.path.join('data', 'scores.csv')
    >>> scores = data_load(scores_fp)
    >>> scores = pass_fail(scores)
    >>> isinstance(scores, pd.DataFrame)
    True
    >>> len(scores.columns)
    3
    >>> scores.loc["Julia", "pass"]=='Yes'
    True
    """
    scores['pass'] = ((scores['highest_score'] >= 50) & 
                      (scores['attempts'] < 3)) | ((scores['highest_score'] >= 70) & 
                                                   (scores['attempts'] < 6)) | ((scores['highest_score'] >= 90) &
                                                                                (scores['attempts'] < 10))
    scores['pass'] = scores['pass'].astype(object).replace({False: 'No', True: 'Yes'})
    return scores
    ...



def av_score(scores):
    """
    returns the average score for those students who passed the test.
    :param scores: dataframe from the second question
    :return: average score
    >>> scores_fp = os.path.join('data', 'scores.csv')
    >>> scores = data_load(scores_fp)
    >>> scores = pass_fail(scores)
    >>> av = av_score(scores)
    >>> isinstance(av, float)
    True
    >>> 91 < av < 92
    True
    """
    return np.mean(scores[scores['pass'] == 'Yes']['highest_score'])
    ...



def highest_score_name(scores):
    """
    finds the highest score and people who received it
    :param scores: dataframe from the second question
    :return: dictionary where the key is the highest score and the value(s) is a list of name(s)
    >>> scores_fp = os.path.join('data', 'scores.csv')
    >>> scores = data_load(scores_fp)
    >>> scores = pass_fail(scores)
    >>> highest = highest_score_name(scores)
    >>> isinstance(highest, dict)
    True
    >>> len(next(iter(highest.items()))[1])
    3
    """
    out = {}
    max_score = np.max(scores['highest_score'])
    maxes = scores[scores['highest_score'] == max_score]
    out[max_score] = maxes.index.tolist()
    return out
    ...


def idx_dup():
    """
    Answers the question in the write up.
    :return:
    >>> ans = idx_dup()
    >>> isinstance(ans, int)
    True
    >>> 1 <= ans <= 6
    True
    """
    return 6
    ...



# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def trick_me():
    """
    Answers the question in the write-up
    :return: a letter
    >>> ans =  trick_me()
    >>> ans == 'A' or ans == 'B' or ans == "C"
    True
    """
    data = [['tom', 'Tom', 10], ['nick', 'Nick', 15], ['juli', 'Juli', 14], ['gary', 'Gary', 10],
            ['brandon', 'Brandon', 15]]
    tricky_1 = pd.DataFrame(data, columns=['Name', 'Name', 'Age'])
    tricky_1.to_csv('tricky_1.csv', index=False)

    tricky_2_fp = os.path.join('tricky_1.csv')
    tricky_2 = pd.read_csv(tricky_2_fp)
    return 'C'
    ...

def reason_dup():
    """
     Answers the question in the write-up
    :return: a letter
    >>> ans =  reason_dup()
    >>> ans == 'A' or ans == 'B' or ans == "C"
    True
    """
    return 'B'
    ...


def trick_bool():
    """
     Answers the question in the write-up
    :return: a list with three letters
    >>> ans =  trick_bool()
    >>> isinstance(ans, list)
    True
    >>> isinstance(ans[1], str)
    True
    """
    return ['D', 'J', 'M']
    ...

def reason_bool():
    """
    Answers the question in the write-up
    :return: a letter
    >>> ans =  reason_bool()
    >>> ans == 'A' or ans == 'B' or ans == "C" or ans =="D"
    True
    """
    return 'B'
    ...


# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


def change(x):
    """
    Returns 'MISSING' when x is `NaN`,
    Otherwise returns x
    >>> change(1.0) == 1.0
    True
    >>> change(np.NaN) == 'MISSING'
    True
    """
    if np.isnan(x):
        return "MISSING"
    else:
        return x
    ...

def correct_replacement(nans):
    """
    changes all np.NaNs to "Missing"
    :param nans: given dataframe
    :return: modified dataframe
    >>> nans = pd.DataFrame([[0,1,np.NaN], [np.NaN, np.NaN, np.NaN], [1, 2, 3]])
    >>> A = correct_replacement(nans)
    >>> (A.values == 'MISSING').sum() == 4
    True
    """
    return nans.applymap(change)
    ...


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


def population_stats(df):
    """
    population_stats which takes in a dataframe df 
    and returns a dataframe indexed by the columns 
    of df, with the following columns:
        - `num_nonnull` contains the number of non-null 
          entries in each column,
        - `pct_nonnull` contains the proportion of entries 
          in each column that are non-null,
        - `num_distinct` contains the number of distinct 
          entries in each column,
        - `pct_distinct` contains the proportion of (non-null) 
          entries in each column that are distinct from each other.
    :Example:
    >>> data = np.random.choice(range(10), size=(100, 4))
    >>> df = pd.DataFrame(data, columns='A B C D'.split())
    >>> out = population_stats(df)
    >>> out.index.tolist() == ['A', 'B', 'C', 'D']
    True
    >>> cols = ['num_nonnull', 'pct_nonnull', 'num_distinct', 'pct_distinct']
    >>> out.columns.tolist() == cols
    True
    >>> (out['num_distinct'] <= 10).all()
    True
    >>> (out['pct_nonnull'] == 1.0).all()
    True
    """
    num_nonnull = df.count()
    pct_nonnull = num_nonnull / len(df)
    num_distinct = df.nunique()
    pct_distinct = num_distinct / num_nonnull
    data = list(zip(num_nonnull, pct_nonnull, num_distinct, pct_distinct))
    out = pd.DataFrame(data, columns=['num_nonnull', 'pct_nonnull', 'num_distinct', 'pct_distinct'],
                       index=df.columns.to_list())
    return out
    ...


def most_common(df, N=10):
    """
    `most_common` which takes in a dataframe df and returns 
    a dataframe of the N most-common values (and their counts) 
    for each column of df.
    :param df: input dataframe.
    :param N: number of most common elements to return (default 10)
.
    :Example:
    >>> data = np.random.choice(range(10), size=(100, 2))
    >>> df = pd.DataFrame(data, columns='A B'.split())
    >>> out = most_common(df, N=3)
    >>> out.index.tolist() == [0, 1, 2]
    True
    >>> out.columns.tolist() == ['A_values', 'A_counts', 'B_values', 'B_counts']
    True
    >>> out['A_values'].isin(range(10)).all()
    True
    """
    out_ts = []
    for i in df.columns:
        a = str(i) + '_values'
        b = str(i) + '_counts'
        out_t = df[i].value_counts().reset_index().set_axis([a, b], axis=1)
        out_ts.append(out_t)
    t = out_ts[0]
    for i in range(len(out_ts))[1:]:
        t = pd.concat([t, out_ts[i]], axis=1)
    return t.head(N)
    ...


# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------


def null_hypoth():
    """
    :Example:
    >>> isinstance(null_hypoth(), list)
    True
    >>> set(null_hypoth()).issubset({1,2,3,4})
    True
    """
    return [1, 3]
    ...


def simulate_null():
    """
    :Example:
    >>> pd.Series(simulate_null()).isin([0,1]).all()
    True
    """
    simulation = np.random.choice([0, 1], p=[0.99, 0.01], size=300)
    return pd.Series(simulation)
    ...


def estimate_p_val(N):
    """
    >>> 0 < estimate_p_val(1000) < 0.1
    True
    """
    results = []
    for _ in range(N):
        simulation = simulate_null()
        sim_false = (simulation == 1).sum()  # test stastistic
        results.append(sim_false)
    return np.count_nonzero(np.array(results) >= 8) / N
    ...


# ---------------------------------------------------------------------
# QUESTION 6
# ---------------------------------------------------------------------


def super_hero_powers(powers):
    """
    `super_hero_powers` takes in a dataframe like 
    powers and returns a list with the following three entries:
        - The name of the super-hero with the greatest number of powers.
        - The name of the most common super-power among super-heroes whose names begin with 'M'.
        - The most popular super-power among those with only one super-power.

    :Example:
    >>> fp = os.path.join('data', 'superheroes_powers.csv')
    >>> powers = pd.read_csv(fp)
    >>> out = super_hero_powers(powers)
    >>> isinstance(out, list)
    True
    >>> len(out)
    3
    >>> all([isinstance(x, str) for x in out])
    True
    """
    hero_power = powers.sum(axis=1).reset_index()
    op = hero_power.sort_values(by=[0], ascending=False).iloc[0, 0]
    cp_m = powers[powers['hero_names'].str.startswith('M')].sum().reset_index()
    cp_m = cp_m.tail(cp_m.shape[0] - 1).sort_values(by=[0], ascending=False).iloc[0, 0]
    sp = powers[hero_power[0] == 1].sum().reset_index()
    sp = sp.tail(sp.shape[0] - 1).sort_values(by=[0], ascending=False).iloc[0, 0]
    return [str(op), cp_m, sp]
    ...


# ---------------------------------------------------------------------
# QUESTION 7
# ---------------------------------------------------------------------


def clean_heroes(heroes):
    """
    clean_heroes takes in the dataframe heroes
    and replaces values that are 'null-value'
    place-holders with np.NaN.

    :Example:
    >>> superheroes_fp = os.path.join('data', 'superheroes.csv')
    >>> heroes = pd.read_csv(superheroes_fp, index_col=0)
    >>> out = clean_heroes(heroes)
    >>> out['Skin color'].isnull().any()
    True
    >>> out['Weight'].isnull().any()
    True
    """
    heroes.loc[heroes['Height'] <= 0, 'Height'] = np.NaN
    heroes.loc[heroes['Weight'] <= 0, 'Weight'] = np.NaN
    for i in heroes.columns:
        heroes.loc[heroes[i] == '-', i] = np.NaN
    return heroes
    ...


def super_hero_stats():
    """
    Returns a list that answers the questions in the notebook.
    :Example:
    >>> out = super_hero_stats()
    >>> out[0] in ['Marvel Comics', 'DC Comics']
    True
    >>> isinstance(out[1], int)
    True
    >>> isinstance(out[2], str)
    True
    >>> out[3] in ['good', 'bad']
    True
    >>> isinstance(out[4], str)
    True
    >>> 0 <= out[5] <= 1
    True
    """
    out = ['Marvel Comics', 558, 'Groot', 'bad', 'Onslaught', 0.2861]
    return out
    ...


# ---------------------------------------------------------------------
# QUESTION 8
# ---------------------------------------------------------------------


def bhbe_col(heroes):
    """
    `bhbe` ('blond-hair-blue-eyes') returns a boolean 
    column that labels super-heroes/villains that 
    are blond-haired *and* blue eyed.

    :Example:
    >>> superheroes_fp = os.path.join('data', 'superheroes.csv')
    >>> heroes = pd.read_csv(superheroes_fp, index_col=0)
    >>> out = bhbe_col(heroes)
    >>> isinstance(out, pd.Series)
    True
    >>> out.dtype == np.dtype('bool')
    True
    >>> out.sum()
    93
    """
    return (heroes['Eye color'] == 'blue') & ((heroes['Hair color'].str.contains('Blond') | (heroes['Hair color'].str.contains('blond'))))
    ...


def observed_stat(heroes):
    """
    observed_stat returns the observed test statistic
    for the hypothesis test.

    :Example:
    >>> superheroes_fp = os.path.join('data', 'superheroes.csv')
    >>> heroes = pd.read_csv(superheroes_fp, index_col=0)
    >>> out = observed_stat(heroes)
    >>> 0.5 <= out <= 1.0
    True
    """
    heroes['bhbe'] = bhbe_col(heroes)
    observed = heroes[heroes['bhbe'] == True]
    observed_proportion = observed[observed['Alignment'] == 'good'].shape[0] / observed.shape[0]
    return observed_proportion
    ...
    

def simulate_bhbe_null(n):
    """
    `simulate_null` that takes in a number `n` 
    that returns a `n` instances of the test statistic 
    generated under the null hypothesis. 
    You should hard code your simulation parameter 
    into the function; the function should *not* read in any data.

    :Example:
    >>> superheroes_fp = os.path.join('data', 'superheroes.csv')
    >>> heroes = pd.read_csv(superheroes_fp, index_col=0)
    >>> out = simulate_bhbe_null(10)
    >>> isinstance(out, pd.Series)
    True
    >>> out.shape[0]
    10
    >>> ((0.45 <= out) & (out <= 1)).all()
    True
    """
    result = []
    p = 0.6757493188010899
    for i in range(n):
        stat = np.random.binomial(93, p) / 93
        result.append(stat)
    return pd.Series(result)
    ...
    

def calc_pval():
    """
    calc_pval returns a list where:
        - the first element is the p-value for 
        hypothesis test (using 100,000 simulations).
        - the second element is Reject if you reject 
        the null hypothesis and Fail to reject if you 
        fail to reject the null hypothesis.

    :Example:
    >>> out = calc_pval()
    >>> len(out)
    2
    >>> 0 <= out[0] <= 1
    True
    >>> out[1] in ['Reject', 'Fail to reject']
    True
    """
    return [0.0015, 'Reject']
    ...

