# lab.py


import pandas as pd
import numpy as np
import os


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def car_null_hypoth():
    """
    Returns a list of valid null hypotheses.
    
    :Example:
    >>> set(car_null_hypoth()) <= set(range(1,11))
    True
    """
    return [1, 4]
    ...


def car_alt_hypoth():
    """
    Returns a list of valid alternative hypotheses.
    
    :Example:
    >>> set(car_alt_hypoth()) <= set(range(1,11))
    True
    """
    return [2, 6]
    ...


def car_test_stat():
    """
    Returns a list of valid test statistics.
    
    :Example:
    >>> set(car_test_stat()) <= set(range(1,5))
    True
    """
    return [1, 4]
    ...


def car_p_value():
    """
    Returns an integer corresponding to the correct explanation.
    
    :Example:
    >>> car_p_value() in [1,2,3,4,5]
    True
    """
    return 5
    ...


# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def clean_apps(df):
    '''
    >>> fp = os.path.join('data', 'googleplaystore.csv')
    >>> df = pd.read_csv(fp)
    >>> cleaned = clean_apps(df)
    >>> len(cleaned) == len(df)
    True
    '''
    cleaned = df.copy()
    cleaned['Size'] = clean_size_helper(cleaned['Size'])
    cleaned['Installs'] = cleaned['Installs'].str.replace(r'\D', '', regex=True).astype('int')

    type_dict = {'Free': '0', 'Paid': '1'}
    cleaned['Type'] = cleaned['Type'].replace(type_dict, regex=True).map(pd.eval).astype('int')
    cleaned['Price'] = cleaned['Price'].str.replace('$', '', regex=False).astype('float')
    cleaned['Last Updated'] = cleaned['Last Updated'].str.strip().str[-4:].astype('int')

    return cleaned
    ...


def clean_size_helper(col):
    repl_dict = {'[kK]': '*1e3', '[mM]': '*1e6', '[bB]': '*1e9', }
    col = col.replace(repl_dict, regex=True).map(pd.eval)
    return col / 1000


def store_info(cleaned):
    '''
    >>> fp = os.path.join('data', 'googleplaystore.csv')
    >>> df = pd.read_csv(fp)
    >>> cleaned = clean_apps(df)
    >>> info = store_info(cleaned)
    >>> len(info)
    4
    >>> info[2] in cleaned.Category.unique()
    True
    '''
    df = cleaned.copy()
    a_1 = df.loc[df['Installs'] >= 100].groupby('Last Updated')['Rating'].median().idxmax()
    a_2 = df.groupby('Content Rating')['Rating'].min().idxmax()
    a_3 = df.groupby('Category')['Price'].mean().idxmax()
    a_4 = df.loc[df['Reviews'] >= 1000].groupby('Category')['Rating'].mean().idxmin()
    return [a_1, a_2, a_3, a_4]
    ...


# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


def std_reviews_by_app_cat(cleaned):
    """
    >>> fp = os.path.join('data', 'googleplaystore.csv')
    >>> play = pd.read_csv(fp)
    >>> cleaned = clean_apps(play)
    >>> out = std_reviews_by_app_cat(cleaned)
    >>> set(out.columns) == set(['Category', 'Reviews'])
    True
    >>> np.all(abs(out.select_dtypes(include='number').mean()) < 10**-7)  # standard units should average to 0!
    True
    """
    df = cleaned.copy()
    grp = df.groupby('Category')['Reviews']
    zscore = lambda x: (x - x.mean()) / x.std()
    df['Reviews'] = grp.transform(zscore)
    return df[['Category', 'Reviews']]
    ...


def su_and_spread():
    """
    >>> out = su_and_spread()
    >>> len(out) == 2
    True
    >>> out[0].lower() in ['medical', 'family', 'equal']
    True
    >>> out[1] in ['ART_AND_DESIGN', 'AUTO_AND_VEHICLES', 'BEAUTY',\
       'BOOKS_AND_REFERENCE', 'BUSINESS', 'COMICS', 'COMMUNICATION',\
       'DATING', 'EDUCATION', 'ENTERTAINMENT', 'EVENTS', 'FINANCE',\
       'FOOD_AND_DRINK', 'HEALTH_AND_FITNESS', 'HOUSE_AND_HOME',\
       'LIBRARIES_AND_DEMO', 'LIFESTYLE', 'GAME', 'FAMILY', 'MEDICAL',\
       'SOCIAL', 'SHOPPING', 'PHOTOGRAPHY', 'SPORTS', 'TRAVEL_AND_LOCAL',\
       'TOOLS', 'PERSONALIZATION', 'PRODUCTIVITY', 'PARENTING', 'WEATHER',\
       'VIDEO_PLAYERS', 'NEWS_AND_MAGAZINES', 'MAPS_AND_NAVIGATION']
    True
    """
    return ['equal', 'GAME', ]
    ...


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


def read_survey(dirname):
    """
    read_survey combines all the survey*.csv files into a singular DataFrame
    :param dirname: directory name where the survey*.csv files are
    :returns: a DataFrame containing the combined survey data
    :Example:
    >>> dirname = os.path.join('data', 'responses')
    >>> out = read_survey(dirname)
    >>> isinstance(out, pd.DataFrame)
    True
    >>> len(out)
    5000
    >>> read_survey('nonexistentfile') # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    FileNotFoundError: ... 'nonexistentfile'
    """
    file_names = os.listdir(dirname)
    file_names = [dirname + '/' + s for s in file_names]
    df_list = [pd.read_csv(f) for f in file_names]

    for i in range(len(df_list)):
        for col in df_list[i].columns:
            if col.lower().startswith('f'):
                df_list[i] = df_list[i].rename({col: 'first name'}, axis=1)
            if col.lower().startswith('l'):
                df_list[i] = df_list[i].rename({col: 'last name'}, axis=1)
            if col.lower().startswith('c'):
                df_list[i] = df_list[i].rename({col: 'current company'}, axis=1)
            if col.lower().startswith('j'):
                df_list[i] = df_list[i].rename({col: 'job title'}, axis=1)
            if col.lower().startswith('e'):
                df_list[i] = df_list[i].rename({col: 'email'}, axis=1)
            if col.lower().startswith('u'):
                df_list[i] = df_list[i].rename({col: 'university'}, axis=1)
    df = pd.concat(df_list, ignore_index=True)
    df = df[['first name', 'last name', 'current company',
             'job title', 'email', 'university']]
    return df
    ...


def com_stats(df):
    """
    com_stats 
    :param df: a DataFrame containing the combined survey data
    :returns: a hardcoded list of answers to the problems in the notebook
    :Example:
    >>> dirname = os.path.join('data', 'responses')
    >>> df = read_survey(dirname)
    >>> out = com_stats(df)
    >>> len(out)
    4
    >>> isinstance(out[0], int)
    True
    >>> isinstance(out[2], str)
    True
    """
    q_1 = df.groupby('current company')['first name'].count().max()
    q_2 = df['email'].fillna('').str.contains(r'\.Edu', case=False).sum()
    max_len = [0, 0]
    for i in df['job title'].unique():
        if len(str(i)) > max_len[0]:
            max_len[0] = len(i)
            max_len[1] = i
    q_3 = max_len[1]
    q_4 = df['job title'].str.contains('manager', case=False).sum()
    return [int(q_1), q_2, q_3, q_4]
    ...


# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------


def combine_surveys(dirname):
    """
    combine_surveys takes in a directory path 
    (containing files favorite*.csv) and combines 
    all of the survey data into one DataFrame, 
    indexed by student ID (a value 0 - 1000).

    :Example:
    >>> dirname = os.path.join('data', 'extra-credit-surveys')
    >>> out = combine_surveys(dirname)
    >>> isinstance(out, pd.DataFrame)
    True
    >>> out.shape
    (1000, 6)
    >>> combine_surveys('nonexistentfile') # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    FileNotFoundError: ... 'nonexistentfile'
    """
    file_names = os.listdir(dirname)
    file_names = [dirname + '/' + s for s in file_names]
    df_list = [pd.read_csv(f) for f in file_names]
    df = df_list[0]
    for i in range(1, len(df_list)):
        df = pd.merge(df, df_list[i], on='id', how='outer')
    df = df.set_index('id')
    return df
    ...


def check_credit(df):
    """
    check_credit takes in a DataFrame with the 
    combined survey data and outputs a DataFrame 
    of the names of students and how many extra credit 
    points they would receive, indexed by their ID (a value 0-1000)

    :Example:
    >>> dirname = os.path.join('data', 'extra-credit-surveys')
    >>> df = combine_surveys(dirname)
    >>> out = check_credit(df)
    >>> out.shape
    (1000, 2)
    """
    df_c = df.copy()
    df_c['self'] = 5 - df_c.isnull().sum(axis=1)
    df_c['credit'] = 0
    for i in df_c.columns:
        num = (df_c[i].isnull().sum()) / len(df_c)
        if num >= 0.9:
            df_c['credit'] = 1
    df_c.loc[df_c['self'] / 5 >= 0.75, 'credit'] += 5
    return df_c[['name', 'credit']]
    ...


# ---------------------------------------------------------------------
# QUESTION 6
# ---------------------------------------------------------------------


def most_popular_procedure(pets, procedure_history):
    """
    What is the most popular Procedure Type for all of the pets we have in our `pets` dataset?
​
    :Example:
    >>> pets_fp = os.path.join('data', 'pets', 'Pets.csv')
    >>> procedure_history_fp = os.path.join('data', 'pets', 'ProceduresHistory.csv')
    >>> pets = pd.read_csv(pets_fp)
    >>> procedure_history = pd.read_csv(procedure_history_fp)
    >>> out = most_popular_procedure(pets, procedure_history)
    >>> isinstance(out,str)
    True
    """
    out = pets.merge(procedure_history, on='PetID')['ProcedureType'].value_counts().sort_values(ascending=False).index[0]
    return out
    ...


def pet_name_by_owner(owners, pets):
    """
    pet names by owner

    :Example:
    >>> owners_fp = os.path.join('data', 'pets', 'Owners.csv')
    >>> pets_fp = os.path.join('data', 'pets', 'Pets.csv')
    >>> owners = pd.read_csv(owners_fp)
    >>> pets = pd.read_csv(pets_fp)
    >>> out = pet_name_by_owner(owners, pets)
    >>> len(out) == len(owners)
    True
    >>> 'Sarah' in out.index
    True
    >>> 'Cookie' in out.values
    True
    """
    pets_and_owner = owners.merge(pets, on='OwnerID')
    pets = pets_and_owner.groupby('OwnerID')['Name_y'].apply(list).apply(lambda x: x[0] if len(x) == 1 else x)
    df = pd.DataFrame(pets).reset_index(level=0)
    df = owners.merge(df, on='OwnerID')
    out = pd.Series(df['Name_y'].values, index=df['Name'])
    return out
    ...


def total_cost_per_city(owners, pets, procedure_history, procedure_detail):
    """
    total cost per city
​
    :Example:
    >>> owners_fp = os.path.join('data', 'pets', 'Owners.csv')
    >>> pets_fp = os.path.join('data', 'pets', 'Pets.csv')
    >>> procedure_detail_fp = os.path.join('data', 'pets', 'ProceduresDetails.csv')
    >>> procedure_history_fp = os.path.join('data', 'pets', 'ProceduresHistory.csv')
    >>> owners = pd.read_csv(owners_fp)
    >>> pets = pd.read_csv(pets_fp)
    >>> procedure_detail = pd.read_csv(procedure_detail_fp)
    >>> procedure_history = pd.read_csv(procedure_history_fp)
    >>> out = total_cost_per_city(owners, pets, procedure_history, procedure_detail)
    >>> set(out.index) <= set(owners['City'])
    True
    """
    df = (owners.merge(pets, on='OwnerID').merge(procedure_history, on='PetID').merge(procedure_detail,
                                                                                      on=['ProcedureSubCode',
                                                                                          'ProcedureType'])).groupby(
        'City').sum()
    out = pd.Series(df['Price'].values, index=df.index)
    return out
    ...
