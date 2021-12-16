# project.py


import pandas as pd
import numpy as np
import os


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def get_assignment_names(grades):
    '''
    get_assignment_names takes in a dataframe like grades and returns 
    a dictionary with the following structure:
    The keys are the general areas of the syllabus: lab, project, 
    midterm, final, disc, checkpoint
    The values are lists that contain the assignment names of that type. 
    For example the lab assignments all have names of the form labXX where XX 
    is a zero-padded two digit number. See the doctests for more details.    
    :Example:
    >>> grades_fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(grades_fp)
    >>> names = get_assignment_names(grades)
    >>> set(names.keys()) == {'lab', 'project', 'midterm', 'final', 'disc', 'checkpoint'}
    True
    >>> names['final'] == ['Final']
    True
    >>> 'project02' in names['project']
    True
    '''
    out = {'lab': [], 'project': [], 'midterm': [], 'final': [], 'disc': [], 'checkpoint': []}
    list_of_column_names = list(grades.columns)
    names = ['lab', 'project', 'midterm', 'final', 'disc']
    for i in names:
        for j in list_of_column_names:
            if j.lower().startswith(i) & ('-' not in j) & ('_' not in j):
                content = out[i] + [j]
                out.update({i: content})
            elif ('checkpoint' in j) & ('-' not in j):
                content = out['checkpoint'] + [j]
                out.update({'checkpoint': content})
    out.update({'checkpoint': list(np.unique(out['checkpoint']))})
    return out
    ...


# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def projects_total(grades):
    '''
    projects_total that takes in grades and computes the total project grade
    for the quarter according to the syllabus. 
    The output Series should contain values between 0 and 1.
    
    :Example:
    >>> grades_fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(grades_fp)
    >>> out = projects_total(grades)
    >>> np.all((0 <= out) & (out <= 1))
    True
    >>> 0.7 < out.mean() < 0.9
    True
    '''
    grades_fill = grades.copy().fillna(0)
    project = get_assignment_names(grades_fill)['project']
    cols = grades_fill.columns.values
    con = []
    score = dict(zip(project, [0] * len(project)))
    y = pd.Series(np.zeros((len(grades_fill),)))
    project_sum = {key: y for key in project}

    for p in project:
        for col in cols:
            if col == p or '{}_free_response'.format(p) == col:
                con.append(col)

            if p in col and '{}_free_response - Max Points'.format(p) == col:
                score[p] = score[p] + grades_fill[col].iloc[0]

            if p in col and '{} - Max Points'.format(p) == col:
                score[p] = score[p] + grades_fill[col].iloc[0]

    for p in project:
        for col in con:
            if p in col:
                project_sum[p] = project_sum[p] + grades_fill.loc[:, col]

    for key in project_sum.keys():
        project_sum[key] = project_sum[key] / score[key]

    return pd.DataFrame(project_sum).fillna(0).sum(axis=1) / len(project)
    ...


# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


def last_minute_submissions(grades):
    """
    last_minute_submissions takes in the dataframe 
    grades and returns a Series indexed by lab assignment that 
    contains the number of submissions that were turned 
    in on time by the student, yet marked 'late' by Gradescope.
    :Example:
    >>> fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(fp)
    >>> out = last_minute_submissions(grades)
    >>> isinstance(out, pd.Series)
    True
    >>> np.all(out.index == ['lab0%d' % d for d in range(1,10)])
    True
    >>> (out > 0).sum()
    8
    """
    grades_fill = grades.copy().fillna(0)
    labs = get_assignment_names(grades_fill)['lab']

    con = []
    for column in grades_fill.columns.values:
        if column.startswith('lab') and column.endswith('Lateness (H:M:S)'):
            con.append(column)
    cols = grades_fill[con]
    out = [0] * len(labs)

    for i in range(len(grades_fill)):
        count = 0
        for col in cols:
            time_h = int(grades_fill[col].iloc[i].split(':')[0])
            time_m = int(grades_fill[col].iloc[i].split(':')[1])
            time_s = int(grades_fill[col].iloc[i].split(':')[2])
            if time_h >= 0 and time_h <= 7 and (time_m > 0 or time_s > 0):
                out[count] = out[count] + 1
            count += 1

    return pd.Series(data=out, index=labs)
    ...


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


def lateness_penalty(col):
    """
    adjust_lateness takes in the dataframe like `grades` 
    and returns a dataframe of lab grades adjusted for 
    lateness according to the syllabus.
    :Example:
    >>> fp = os.path.join('data', 'grades.csv')
    >>> col = pd.read_csv(fp)['lab01 - Lateness (H:M:S)']
    >>> out = lateness_penalty(col)
    >>> isinstance(out, pd.Series)
    True
    >>> set(out.unique()) <= {1.0, 0.9, 0.7, 0.4}
    True
    """
    grades_fill = col.copy().fillna(0)
    out = []
    for i in grades_fill:
        time_h = int(i.split(':')[0])
        if time_h <= 6:
            out.append(1.0)
        elif time_h < 24 * 7:
            out.append(0.9)
        elif time_h < 24 * 14:
            out.append(0.7)
        else:
            out.append(0.4)
        # else:
        #     out.append(0)
    return pd.Series(out)
    ...


# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------


def process_labs(grades):
    """
    process_labs that takes in a dataframe like grades and returns
    a dataframe of processed lab scores. The output should:
      * share the same index as grades,
      * have columns given by the lab assignment names (e.g. lab01,...lab10)
      * have values representing the lab grades for each assignment, 
        adjusted for Lateness and scaled to a score between 0 and 1.
    :Example:
    >>> fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(fp)
    >>> out = process_labs(grades)
    >>> out.columns.tolist() == ['lab%02d' % x for x in range(1,10)]
    True
    >>> np.all((0.65 <= out.mean()) & (out.mean() <= 0.90))
    True
    """
    grades_fill = grades.copy()
    labs = get_assignment_names(grades_fill)['lab']
    grades_fill['pel'] = grades_fill[labs[0]]
    for i in labs:
        lateness = i + ' - Lateness (H:M:S)'
        max_p = i + ' - Max Points'
        grades_fill['pel'] = lateness_penalty(grades_fill[lateness])
        grades_fill[i] = (grades_fill[i] / grades_fill[max_p]) * grades_fill['pel']

    out = grades_fill[labs]
    return out
    ...


# ---------------------------------------------------------------------
# QUESTION 6
# ---------------------------------------------------------------------


def lab_total(processed):
    """
    lab_total takes in dataframe of processed assignments (like the output of 
    Question 5) and computes the total lab grade for each student according to
    the syllabus (returning a Series). 
    
    Your answers should be proportions between 0 and 1.
    :Example:
    >>> cols = 'lab01 lab02 lab03'.split()
    >>> processed = pd.DataFrame([[0.2, 0.90, 1.0]], index=[0], columns=cols)
    >>> np.isclose(lab_total(processed), 0.95).all()
    True
    """
    grade_fill = processed.fillna(0)
    min_grade = grade_fill.min(axis=1)
    out = (grade_fill.sum(axis=1) - min_grade) / (len(grade_fill.columns) - 1)
    return out
    ...


# ---------------------------------------------------------------------
# QUESTION 7
# ---------------------------------------------------------------------


def total_points(grades):
    """
    total_points takes in grades and returns the final
    course grades according to the syllabus. Course grades
    should be proportions between zero and one.
    :Example:
    >>> fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(fp)
    >>> out = total_points(grades)
    >>> np.all((0 <= out) & (out <= 1))
    True
    >>> 0.7 < out.mean() < 0.9
    True
    """
    grades_fill = grades.copy().fillna(0)
    lab_score = lab_total(process_labs(grades_fill))
    proj_score = projects_total(grades_fill)

    check_score = total_score_helper('checkpoint', grades_fill)
    disc_score = total_score_helper('disc', grades_fill)
    mid_score = total_score_helper('midterm', grades_fill)
    final_score = total_score_helper('final', grades_fill)

    out = lab_score * 0.2 + proj_score * 0.3 + check_score * 0.025 + disc_score * 0.025 + mid_score * 0.15 + final_score * 0.3
    return out
    ...


def total_score_helper(taskname, grades_fill):
    task = get_assignment_names(grades_fill)[taskname]
    for i in task:
        max_p = i + ' - Max Points'
        grades_fill[i] = grades_fill[i] / grades_fill[max_p]
    task_score = grades_fill[task].sum(axis=1) / len(task)
    return task_score


def final_grades(total):
    """
    final_grades takes in the final course grades
    as above and returns a Series of letter grades
    given by the standard cutoffs.
    :Example:
    >>> out = final_grades(pd.Series([0.92, 0.81, 0.41]))
    >>> np.all(out == ['A', 'B', 'F'])
    True
    """
    total_fill = total.copy().fillna(0)
    out = []
    for i in total_fill:
        if i >= 0.9:
            out.append('A')
        elif i >= 0.8:
            out.append('B')
        elif i >= 0.7:
            out.append('C')
        elif i >= 0.6:
            out.append('D')
        else:
            out.append('F')
    return pd.Series(out)
    ...


def letter_proportions(grades):
    """
    letter_proportions takes in the dataframe grades 
    and outputs a Series that contains the proportion
    of the class that received each grade.
    :Example:
    >>> fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(fp)
    >>> out = letter_proportions(grades)
    >>> np.all(out.index == ['B', 'C', 'A', 'D', 'F'])
    True
    >>> out.sum() == 1.0
    True
    """
    grades_fill = grades.copy().fillna(0)
    final_letter = final_grades(total_points(grades_fill))
    final_count = final_letter.value_counts()
    final_p = final_count / final_count.sum()
    return final_p
    ...


# ---------------------------------------------------------------------
# QUESTION 8
# ---------------------------------------------------------------------


def simulate_pval(grades, N):
    """
    simulate_pval takes in the number of
    simulations N and grades and returns
    the likelihood that the grade of seniors
    was worse than the class under null hypothesis conditions
    (i.e. calculate the p-value).
    :Example:
    >>> fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(fp)
    >>> out = simulate_pval(grades, 1000)
    >>> 0 <= out <= 0.1
    True
    """
    scores = total_points(grades)
    mapping = {'SR': 'senior', 'SO': 'rest', 'JR': 'rest'}
    df = pd.DataFrame({'Level': grades['Level']. map(mapping), 'score': scores})
    observe = df.groupby(by = 'Level').mean().loc['senior', :][0]
    mean_scores = []
    for _ in range(N):
        sample = df.sample(len(grades), replace=True).mean()[0]
        mean_scores .append(sample)
    out = np.count_nonzero(mean_scores<=observe) / N
    return out
    ...


# ---------------------------------------------------------------------
# QUESTION 9
# ---------------------------------------------------------------------


def total_points_with_noise(grades):
    """
    total_points_with_noise takes in a dataframe like grades, 
    adds noise to the assignments as described in notebook, and returns
    the total scores of each student calculated with noisy grades.
    :Example:
    >>> fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(fp)
    >>> out = total_points_with_noise(grades)
    >>> np.all((0 <= out) & (out <= 1))
    True
    >>> 0.7 < out.mean() < 0.9
    True
    """
    grades_fill = grades.copy().fillna(0)
    grades_fill = noise_score_helper(grades_fill)

    return total_points(grades_fill)

    ...

def noise_score_helper(grades_fill):
    con = []
    out = grades_fill.copy()
    for i in ['lab', 'project', 'midterm', 'final', 'disc', 'checkpoint']:
        con.append(get_assignment_names(grades_fill)[i])
    for i in con:
        for j in i:
            noise = pd.Series(np.random.normal(0, 0.02, size=(len(grades_fill), )))
            max_p = j + ' - Max Points'
            out[j] = np.clip((grades_fill[j] + noise), 0, grades_fill[max_p])
    return out


# ---------------------------------------------------------------------
# QUESTION 10
# ---------------------------------------------------------------------


def short_answer():
    """
    short_answer returns (hard-coded) answers to the 
    questions listed in the notebook. The answers should be
    given in a list with the same order as questions.
    :Example:
    >>> out = short_answer()
    >>> len(out) == 5
    True
    >>> len(out[2]) == 2
    True
    >>> 50 < out[2][0] < 100
    True
    >>> 0 < out[3] < 1
    True
    >>> isinstance(out[4][0], bool)
    True
    >>> isinstance(out[4][1], bool)
    True
    """
    a = 'The distribution of the difference between the noise sample and the original tasks scores can be plot as a curve of approximately normal distribution with an average value of about 0 and a range of about [-0.0004,0.0004].'
    b= 64
    c = [-0.00019585801165846183, 0.00024811503861455473]
    d = 0.01
    e= [True, True]
    return [a, b, c, d, e]

    ...
