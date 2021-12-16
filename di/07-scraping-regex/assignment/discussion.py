# discussion.py


import os
import numpy as np
import pandas as pd
import requests
import time
from bs4 import BeautifulSoup
import re


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def get_website_urls():
    """
    Get all the website URLs

    :Example:
    >>> urls = get_website_urls()
    >>> len(urls)
    50
    >>> 'catalogue' in urls[0]
    True
    """
    out = [f'http://books.toscrape.com/catalogue/page-{num}.html' for num in range(1,51)]
    return out
    ...


# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def book_categories():
    """
    Get all the book categories and return them as a list

    :Example:
    >>> categories = book_categories()
    >>> len(categories)
    50
    >>> 'Classics' in categories
    True
    """
    book_request = requests.get('http://books.toscrape.com/index.html')
    soup = BeautifulSoup(book_request.text)

    sidebar = soup.find_all('div', {'class': 'side_categories'})[0]
    result = []
    for cat in sidebar.find_all('a'):
        result.append(cat.text)
    result = [i.strip() for i in result]
    return result[1:]
    ...


# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


def duplicate_words(s):
    """
    Provide a list of all words that are duplicates in an input sentence.
    Assume that the sentences are lower case.

    :Example:
    >>> duplicate_words('let us plan for a horror movie movie this weekend')
    ['movie']
    >>> duplicate_words('I like surfing')
    []
    >>> duplicate_words('the class class is good but the tests tests are hard')
    ['class', 'tests']
    """
    return re.findall(r'(\b\w+\b)\s+\b\1\b', s)
    ...


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


def laptop_details(df):
    """
    Given a df with product description - Return df with added columns of 
    processor (i3, i5), generation (9th Gen, 10th Gen), 
    storage (512 GB SSD, 1 TB HDD), display_in_inch (15.6 inch, 14 inch)

    :Example:
    >>> df = pd.read_csv('data/laptop_details.csv')
    >>> new_df = laptop_details(df)
    >>> new_df.shape
    (21, 5)
    >>> new_df['processor'].nunique()
    3
    """
    df_c = df.copy()
    df_c['processor'] = df_c['laptop_description'].str.extract(r'(\bi\d+\b)')
    df_c['generation'] = df_c['laptop_description'].str.extract(r'(\b\d+(th|nd|rd)\b\s+\bGen\b)')[0]
    df_c['storage'] = df_c['laptop_description'].str.extract(r'(\b\d+\s(GB|TB)\s(SSD|HDD)\b)')[0]
    df_c['display_inch'] = df_c['laptop_description'].str.extract(r'(\b\d+.?\d+\s+inch\b)')[0]
    return df_c
    ...
