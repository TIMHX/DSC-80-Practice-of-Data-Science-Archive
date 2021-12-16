# project.py


import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def is_monotonic(arr):
    """
    Given a numpy array of numbers, determines if each entry is >= than the previous.
    
    Example
    -------
    
    >>> is_monotonic(np.array([3, 6, 2, 8]))
    False
    
    """
    dif = np.diff(arr)
    return np.all(dif >= 0)
    ...


def monotonic_by_country(vacs):
    """
    Given a dataframe like `vacs`, returns a dataframe with one row for each country/region and 
    three bool columns -- Doses_admin_monotonic, People_partially_vaccinated_monotonic, and 
    People_fully_vaccinated_monotonic. An entry in the Doses_admin column should be True if the 
    country's Doses_admin is monotonically increasing and False otherwise; likewise for the other
    columns. The index of the returned dataframe should contain country names.
    
    Example
    -------
    
    >>> vacs = pd.read_csv('data/covid-vaccinations-subset.csv')
    >>> result = monotonic_by_country(vacs)
    >>> result.loc['Venezuela', 'Doses_admin_monotonic']
    False
    
    """
    group = vacs.groupby('Country_Region')
    col1 = group.Doses_admin.agg(is_monotonic)
    col2 = group.People_partially_vaccinated.agg(is_monotonic)
    col3 = group.People_fully_vaccinated.agg(is_monotonic)

    out = pd.concat([col1, col2, col3], axis=1).set_axis([
        'Doses_admin_monotonic', 'People_partially_vaccinated_monotonic',
        'People_fully_vaccinated_monotonic'
    ],
        axis=1,
        inplace=False)
    return out
    ...


# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def robust_totals(vacs):
    """
    Accepts a dataframe like vacs above and returns a dataframe with one row for each 
    country/region and three columns -- Doses_admin, People_partially_vaccinated, and 
    People_fully_vaccinated -- where an entry in the Doses_admin column is the 97th 
    percentile of the values in that column for that country; likewise for the other 
    columns. The index of the returned dataframe should contain country names.
    
    Example
    -------
    
    >>> vacs = pd.read_csv('data/covid-vaccinations-subset.csv')
    >>> tots = robust_totals(vacs)
    >>> int(tots.loc['Venezuela', 'Doses_admin'])
    15714857
    
    """
    group = vacs.groupby('Country_Region')
    col1 = group.Doses_admin.agg(q97)
    col2 = group.People_partially_vaccinated.agg(q97)
    col3 = group.People_fully_vaccinated.agg(q97)

    out = pd.concat([col1, col2, col3], axis=1).set_axis([
        'Doses_admin', 'People_partially_vaccinated',
        'People_fully_vaccinated'
    ],
        axis=1,
        inplace=False)
    return out
    ...


def q97(x):
    return np.percentile(x, 97)


# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


def fix_dtypes(pop_raw):
    """
    Accepts a dataframe like pop_raw above and returns a data frame with exactly
    the same columns and rows, but with the data types "fixed" to be appropriate
    for the data contained within. In addition, ensure that all missing values are
    represented by NaNs. All percentages should be represented as decimals -- e.g.,
    27% should be 0.27.
    
    Example
    -------
    
    >>> pops_raw = pd.read_csv('data/populations.csv')
    >>> pops = fix_dtypes(pops_raw)
    >>> pops.loc[pops['Country (or dependency)'] == 'Montserrat', 'Population (2020)'].iloc[0]
    4993
    
    """
    pop_c = pop_raw.copy()
    for i in pop_c.columns[1:]:
        sample = pop_c[i][0]
        if type(sample) is str:
            if '.' in sample or ' %' in sample:
                pop_c[i] = pop_c[i].replace('N.A.', np.nan)
                pop_c[i] = pop_c[i].str.replace(',', '')
                if ' %' in sample:
                    pop_c[i] = pop_c[i].str.replace(' %', '')
                    pop_c[i] = pop_c[i].astype('float')
                    pop_c[i] = pop_c[i] * 0.01
                pop_c[i] = pop_c[i].astype('float')
            else:
                pop_c[i] = pop_c[i].replace('N.A.', np.nan)
                pop_c[i] = pop_c[i].str.replace(',', '')
                if pop_c[i].isna().sum() > 0:
                    pop_c[i] = pop_c[i].astype('float')
                else:
                    pop_c[i] = pop_c[i].astype('int')
        else:
            pop_c[i] = pop_c[i].astype('int')
    return pop_c
    ...


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


def missing_in_pops(tots, pop):
    """
    takes in two tables, the first, like tots above, containing the total number of
    vaccinations per country/region, and the second like pops above, containing the
    population of each country/region. It should return a Python set of names that appear
    in tots but not in pops.
    
    Example
    -------
    >>> tots = pd.DataFrame({
    ...         'Doses_admin': [1, 2, 3],
    ...         'People_partially_vaccinated': [1, 2, 3],
    ...         'People_fully_vaccinated': [1, 2, 3]
    ...     },
    ...     index = ['China', 'Angola', 'Republic of Data Science']
    ... )
    >>> pops_raw = pd.read_csv('data/populations.csv')
    >>> pops = fix_dtypes(pops_raw)
    >>> missing_in_pops(tots, pops)
    {'Republic of Data Science'}
    """
    return set(tots.index[tots.index.isin(pop['Country (or dependency)']) == False])
    ...


def fix_names(pops):
    """
    Accepts one argument -- a table like pops -- and returns a copy of pops, but with the 
    "Country (or dependency)" column changed so that all countries that appear in tots 
    also appear in the result, with a few exceptions listed in the notebook.
    
    Example
    -------
    
    >>> pops_raw = pd.read_csv('data/populations.csv')
    >>> pops = fix_dtypes(pops_raw)
    >>> fixed = fix_names(pops)
    >>> 'Burma' in fixed['Country (or dependency)'].values
    True
    >>> 'Myanmar' in fixed['Country (or dependency)'].values
    False
    
    """
    pop_c = pops.copy()
    mapping = {
        'Myanmar': 'Burma',
        "Côte d'Ivoire": "Cote d'Ivoire",
        'Czech Republic (Czechia)': 'Czechia',
        'South Korea': 'Korea, South',
        'Saint Kitts & Nevis': 'Saint Kitts and Nevis',
        'St. Vincent & Grenadines': 'Saint Vincent and the Grenadines',
        'Sao Tome & Principe': 'Sao Tome and Principe',
        'United States': 'US'
    }
    for i in mapping.keys():
        pop_c.loc[pop_c['Country (or dependency)'] == i,
                  'Country (or dependency)'] = mapping[i]
    return pop_c
    ...


# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------


def top_k_fully_vaccinated(tots, pops_fixed, k):
    """
    Accepts three arguments: a dataframe like tots, a dataframe like pops_fixed, and a
    number, k, and returns a pandas Series of the  𝑘  ten vaccination rates (number of
    fully vaccinated divided by total population) of any country/region, sorted in
    descending order. The index of the Series should be the country/region name,
    and the rates should be decimal numbers between 0 and 1.
    
    Example
    -------
    
    >>> # this file contains a subset of `tots`
    >>> tots_sample = pd.read_csv('data/tots_sample_for_tests.csv').set_index('Country_Region')
    >>> pops = fix_dtypes(pd.read_csv('data/populations.csv'))
    >>> top_k_fully_vaccinated(tots_sample, pops, 3).index[2]
    'Oman'
    """
    tots_c = tots.copy()
    pop_c = pops_fixed.copy()
    full = pd.DataFrame(tots_c['People_fully_vaccinated'])
    pop = pop_c[['Country (or dependency)', 'Population (2020)']]
    pop = pop.merge(full,
                    left_on='Country (or dependency)',
                    right_on='Country_Region',
                    how='inner')
    pop['Rate'] = pop['People_fully_vaccinated'] / pop['Population (2020)']
    pop.sort_values('Rate', ascending=False, inplace=True)
    pop.drop(['Population (2020)', 'People_fully_vaccinated'],
             axis=1,
             inplace=True)
    pop.columns = ['Country', 'Rate']
    pop.set_index('Country', inplace=True)
    return pop.head(10)['Rate']
    ...


# ---------------------------------------------------------------------
# QUESTION 6
# ---------------------------------------------------------------------


def clean_israel_data(df):
    """
    Accepts a data frame like israel_raw and returns a new dataframe where the missing
    values are replaced by NaNs and the "Age" column's data type is float. Furthermore,
    the "Vaccinated" and "Severe Sickness" columns should be stored as bools. The shape
    of the returned data frame should be the same as israel_raw, and, as usual, your
    function should not modify the input argument.
    
    Example
    -------
    
    >>> israel_raw = pd.read_csv('data/israel-subset.csv')
    >>> result = clean_israel_data(israel_raw)
    >>> str(result.dtypes['Age'])
    'float64'
    
    """
    d = df.copy()
    d = d.replace('-', np.NaN)
    d['Age'] = d['Age'].astype('float')
    d['Vaccinated'] = d['Vaccinated'].astype('boolean')
    d['Severe Sickness'] = d['Severe Sickness'].astype('boolean')
    return d
    ...


# ---------------------------------------------------------------------
# QUESTION 7
# ---------------------------------------------------------------------


def mcar_permutation_tests(israel, n_permutations=100):
    """
    Accepts two arguments -- a dataframe like israel and a number n_permutations of
    permutations -- and runs the two permutation tests described in the notebook. Your
    function should return a 2-tuple where the first entry is an array of the simulated test
    statistics for the first permutation test, and the second entry is an array of
    simulated test statistics for the second permutation test.
    
    Example
    -------
    
    >>> israel_raw = pd.read_csv('data/israel-subset.csv')
    >>> israel = clean_israel_data(israel_raw)
    >>> res = mcar_permutation_tests(israel, n_permutations=3)
    >>> isinstance(res[0], np.ndarray) and isinstance(res[1], np.ndarray)
    True
    >>> len(res[0]) == len(res[1]) == 3 # because only 3 permutations
    True
    
    """
    # israel_c = df.copy()
    # is_vaccinated = israel_c['Vaccinated'].values
    # is_sick = israel_c['Severe Sickness'].values
    # Age = israel_c['Age'].isna().values
    #
    # n_vaccinated = np.count_nonzero(is_vaccinated)
    # n_non_vaccinated = len(is_vaccinated) - n_vaccinated
    # n_sick = np.count_nonzero(is_sick)
    # n_non_sick = len(is_sick) - n_sick
    #
    # del (israel_c)
    # is_vaccinated_permutations = np.column_stack([
    #     np.random.permutation(is_vaccinated).astype('bool') for _ in range(n_permutations)
    # ]).T
    #
    # mean_vaccinated = (Age * is_vaccinated_permutations).sum(axis=1) / n_vaccinated
    # mean_non_vaccinated = (Age * ~is_vaccinated_permutations).sum(
    #     axis=1) / n_non_vaccinated
    # vaccinated_differences = abs(mean_non_vaccinated - mean_vaccinated)
    # del (is_vaccinated_permutations)
    # del (mean_vaccinated)
    # del (mean_non_vaccinated)
    #
    # is_sick_permutations = np.column_stack([
    #     np.random.permutation(is_sick).astype('bool') for _ in range(n_permutations)
    # ]).T
    # mean_sick = (Age * is_sick_permutations).sum(axis=1) / n_sick
    # mean_non_sick = (Age * ~is_sick_permutations).sum(
    #     axis=1) / n_non_sick
    # sick_differences = abs(mean_non_sick - mean_sick)
    #
    # return (vaccinated_differences, sick_differences)
    l1 = []
    l2 = []
    for i in range(n_permutations):
        # shuffle the gender column
        shuffled_col = (
            israel['Age']
                .sample(replace=False, frac=1)
                .reset_index(drop=True)
        )
        # print(shuffled_col)
        # put them in a table
        df = israel.copy()
        df['Age'] = shuffled_col
        # print(df)
        notVaxed = df.groupby(df['Age'].isna()).mean()['Vaccinated'][0]
        vaxed = df.groupby(df['Age'].isna()).mean()['Vaccinated'][1]
        notSick = df.groupby(df['Age'].isna()).mean()['Severe Sickness'][0]
        sick = df.groupby(df['Age'].isna()).mean()['Severe Sickness'][1]
        l1.append(notVaxed - vaxed)
        l2.append(notSick - sick)
    return (np.array(l1), np.array(l2))
    ...


def missingness_type():
    """
    Returns a single integer corresponding to the option below that you think describes
    the type of missingess in this data:

        1. MCAR (Missing completely at random)
        2. MAR (Missing at random)
        3. NMAR (Not missing at random)
        4. Missing by design
        
    Example
    -------
    >>> missingness_type() in {1, 2, 3, 4}
    True
    
    """
    return 3


# ---------------------------------------------------------------------
# QUESTION 8
# ---------------------------------------------------------------------


def effectiveness(vax):
    """
    Accepts a dataframe like vax above, and returns the effectiveness of the
    vaccine against severe illness.
    
    Example
    -------
    
    >>> example_vax = pd.DataFrame({
    ...             'Age': [15, 20, 25, 30, 35, 40],
    ...             'Vaccinated': [True, True, True, False, False, False],
    ...             'Severe Sickness': [True, False, False, False, True, True]
    ...         })
    >>> effectiveness(example_vax)
    0.5
    
    """
    # vax_c = df.copy()
    # vax_g = vax_c.groupby(['Vaccinated', 'Severe Sickness']).count().reset_index()
    # p_1 = vax_g.loc[1].Age / (vax_g.loc[0].Age + vax_g.loc[1].Age)
    # p_2 = vax_g.loc[3].Age / (vax_g.loc[3].Age + vax_g.loc[2].Age)
    # return 1 - p_2 / p_1
    vaxed = vax[vax['Vaccinated']].shape[0]
    sickAndVaxed = vax[vax['Vaccinated'] & vax['Severe Sickness']].shape[0]
    pv = sickAndVaxed / vaxed
    unvaxed = vax[vax['Vaccinated'] == False].shape[0]
    sickAndUnvaxed = vax[(vax['Vaccinated'] == False) & (vax['Severe Sickness'] == True)].shape[0]
    pu = sickAndUnvaxed / unvaxed
    effectiveness = (pu - pv) / pu
    return effectiveness
    ...


# ---------------------------------------------------------------------
# QUESTION 9
# ---------------------------------------------------------------------


AGE_GROUPS = [
    '12-15',
    '16-19',
    '20-29',
    '30-39',
    '40-49',
    '50-59',
    '60-69',
    '70-79',
    '80-89',
    '90-'
]


def stratified_effectiveness(df):
    """
    Accepts one argument -- a dataframe like vax -- and returns the effectiveness of the
    vaccine within each of the age groups in AGE_GROUPS. The return value of the function
    should be a Series of the same length as AGE_GROUPS, with the index of the series being
    the age group as a string.
    
    Example
    -------
    
    >>> vax_subset = pd.read_csv('data/vax_subset_for_tests.csv').astype({'Vaccinated': bool, 'Severe Sickness': bool})
    >>> stratified_effectiveness(vax_subset).index[0]
    '12-15'
    >>> len(stratified_effectiveness(vax_subset))
    10
    
    """
    bins = [12, 16, 20, 30, 40, 50, 60, 70, 80, 90, float('inf')]
    cols = pd.cut(df['Age'], bins=bins, labels=AGE_GROUPS, right=False)
    df_c = df.copy()
    df_c['Age_group'] = cols
    return df_c.groupby(by='Age_group').apply(effectiveness)
    ...


# ---------------------------------------------------------------------
# QUESTION 10
# ---------------------------------------------------------------------


def effectiveness_calculator(
        *,
        young_vaccinated_pct,
        old_vaccinated_pct,
        young_risk_vaccinated,
        young_risk_unvaccinated,
        old_risk_vaccinated,
        old_risk_unvaccinated
):
    """Given various vaccination probabilities, computes the effectiveness.
    
    See the notebook for full instructions.
    
    Example
    -------
    >>> test_eff = effectiveness_calculator(
    ...  young_vaccinated_pct=.5,
    ...  old_vaccinated_pct=.5,
    ...  young_risk_vaccinated=.01,
    ...  young_risk_unvaccinated=.20,
    ...  old_risk_vaccinated=.01,
    ...  old_risk_unvaccinated=.20
    ... )
    >>> test_eff['Overall'] == test_eff['Young'] == test_eff['Old'] == 0.95
    True
    
    """
    young = (young_risk_unvaccinated - young_risk_vaccinated) / young_risk_unvaccinated
    old = (old_risk_unvaccinated - old_risk_vaccinated) / old_risk_unvaccinated
    pv = (young_vaccinated_pct * young_risk_vaccinated + old_vaccinated_pct * old_risk_vaccinated) / (
                young_vaccinated_pct + old_vaccinated_pct)
    pu = ((1 - young_vaccinated_pct) * young_risk_unvaccinated + (1 - old_vaccinated_pct) * old_risk_unvaccinated) / (
                (1 - young_vaccinated_pct) + (1 - old_vaccinated_pct))
    overall = (pu - pv) / pu
    return {'Overall': overall, 'Young': young, 'Old': old}
    ...


# ---------------------------------------------------------------------
# QUESTION 11
# ---------------------------------------------------------------------


def extreme_example():
    """
    Accepts no arguments and returns a dictionary whose keys are the arguments to 
    the function effectiveness_calculator above. When your function is called and 
    the dictionary is passed to effectiveness_calculator, it should return an 
    Overall effectiveness that is negative and a Young and Old effectiveness 
    of >80% each.
    
    Example
    -------
    
    >>> isinstance(extreme_example(), dict)
    True
    >>> keys = {
    ... 'old_risk_unvaccinated',
    ... 'old_risk_vaccinated',
    ... 'old_vaccinated_pct',
    ... 'young_risk_unvaccinated',
    ... 'young_risk_vaccinated',
    ... 'young_vaccinated_pct'
    ... }
    >>> extreme_example().keys() == keys
    True
    """
    return {'young_vaccinated_pct' : .01,
    'old_vaccinated_pct' : .99,
    'young_risk_vaccinated' : .01,
    'young_risk_unvaccinated' : .06,
    'old_risk_vaccinated' : .10,
    'old_risk_unvaccinated' : .50}
    ...
