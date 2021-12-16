# lab.py


import os
import pandas as pd
import numpy as np
import requests
import bs4


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def question1():
    """
    NOTE: You do NOT need to do anything with this function.
    The function for this question makes sure you
    have a correctly named HTML file in the right
    place. Note: This does NOT check if the supplementary files
    needed for your page are there!
    >>> question1()
    >>> os.path.exists('lab06_1.html')
    True
    """
    # Don't change this function body!
    # No python required; create the HTML file.

    return


# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def extract_book_links(text):
    """
    :Example:
    >>> fp = os.path.join('data', 'products.html')
    >>> out = extract_book_links(open(fp, encoding='utf-8').read())
    >>> url = 'scarlet-the-lunar-chronicles-2_218/index.html'
    >>> out[1] == url
    True
    """
    soup = bs4.BeautifulSoup(text, features="lxml")
    cell = soup.find_all('article', {'class': 'product_pod'})
    result = []
    for book in cell:
        price = book.find('p', {'class': 'price_color'}).text
        rate = book.select('p[class*="star-rating "]')[0].get('class')[1]
        if (float(price[2:]) < 50) and ((rate == 'Five') | (rate == 'Four')):
            hrefs = book.find_all('a')
            result.append(hrefs[0].get('href'))
    return result
    ...


def get_product_info(text, categories):
    """
    :Example:
    >>> fp = os.path.join('data', 'Frankenstein.html')
    >>> out = get_product_info(open(fp, encoding='utf-8').read(), ['Default'])
    >>> isinstance(out, dict)
    True
    >>> 'Category' in out.keys()
    True
    >>> out['Rating']
    'Two'
    """
    soup = bs4.BeautifulSoup(text, features="lxml")
    category = soup.select('a[href*="/category/books/"]')[0].text
    if category not in categories:
        return None

    result = {}
    availability = soup.find('p', {'class': 'instock availability'}).text.strip()

    for i in soup.find_all('meta'):
        if i.get('name') == 'description':
            description = i.get('content').strip()

    n_review = soup.find("th", text="Number of reviews").find_next_sibling("td").text
    price_ex = soup.find("th", text="Price (excl. tax)").find_next_sibling("td").text
    price_in = soup.find("th", text="Price (incl. tax)").find_next_sibling("td").text
    product_t = soup.find("th", text="Product Type").find_next_sibling("td").text
    star = soup.select('p[class*="star-rating "]')[0].get('class')[1]
    tax = soup.find("th", text="Tax").find_next_sibling("td").text
    title = soup.find("h1").text
    upc = soup.find("th", text="UPC").find_next_sibling("td").text

    result['Availability'] = availability
    result['Category'] = category
    result['Description'] = description
    result['Number of reviews'] = n_review
    result['Price (excl. tax)'] = price_ex
    result['Price (incl. tax)'] = price_in
    result['Product Type'] = product_t
    result['Rating'] = star
    result['Tax'] = tax
    result['Title'] = title
    result['UPC'] = upc
    return result
    ...


def scrape_books(k, categories):
    """
    :param k: number of book-listing pages to scrape.
    :returns: a dataframe of information on (certain) books
    on the k pages (as described in the question).
    :Example:
    >>> out = scrape_books(1, ['Mystery'])
    >>> out.shape
    (1, 11)
    >>> out['Rating'][0] == 'Four'
    True
    >>> out['Title'][0] == 'Sharp Objects'
    True
    """
    pages = [f'http://books.toscrape.com/catalogue/page-{num}.html' for num in range(1, k + 1)]
    book_list = []
    book_dict = []
    for page in pages:
        page_request = requests.get(page)
        book_list.extend(extract_book_links(page_request.text))
    for book in book_list:
        book_request = requests.get('http://books.toscrape.com/catalogue/' + book)
        book_info = get_product_info(book_request.text, categories)
        if book_info is not None:
            book_dict.append(book_info)
    return pd.DataFrame(book_dict)
    ...


# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


def stock_history(ticker, year, month):
    """
    Given a stock code and month, return the stock price details for that month
    as a dataframe

    >>> history = stock_history('BYND', 2019, 6)
    >>> history.shape == (20, 13)
    True
    >>> history.label.iloc[-1]
    'June 03, 19'
    """
    url = 'https://financialmodelingprep.com/api/v3/historical-price-full/{t}?apikey=a2311ec4aaf814e3bb089cd297376cff'.format(
        t=ticker)
    data = pd.read_json(url)['symbol']
    flatten = pd.json_normalize(pd.read_json(url)['historical'])
    df = pd.concat([data, flatten], join='outer', axis=1)
    df['date'] = pd.to_datetime(df['date'])
    df_filtered = df[(df['date'].dt.month == month)]
    df_filtered = df_filtered[(df_filtered['date'].dt.year == year)]
    df_filtered = df_filtered[df_filtered['symbol'] == ticker]
    out = df_filtered.reset_index().drop(['index', 'symbol'], axis=1)
    return out
    ...


def stock_stats(history):
    """
    Given a stock's trade history, return the percent change and transactions
    in billion dollars.

    >>> history = stock_history('BYND', 2019, 6)
    >>> stats = stock_stats(history)
    >>> len(stats[0]), len(stats[1])
    (7, 6)
    >>> float(stats[0][1:-1]) > 30
    True
    >>> float(stats[1][:-1]) > 1
    True
    """
    df = history.copy()
    df = df.sort_values('date').reset_index()
    op = df.iloc[0].open
    cl = df.iloc[-1].close
    PC = str(round((cl - op) / op * 100, 2)) + '%'
    if not PC.startswith('-'):
        PC = '+' + PC
    TTV = str(round(sum((history['low'] + history['high']) / 2 * history['volume'] / (10 ** 9)), 2)) + 'B'
    return (PC, TTV)
    ...


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


def get_comments(storyid):
    """
    Returns a dataframe of all the comments below a news story
    >>> out = get_comments(18344932)
    >>> out.shape
    (18, 5)
    >>> out.loc[5, 'by']
    'RobAtticus'
    >>> out.loc[5, 'time'].day
    31
    """
    import json
    out = []
    url = "https://hacker-news.firebaseio.com/v0/item/{}.json?print=pretty".format(storyid)
    kids = json.loads(requests.get(url).text)['kids']

    def DSF():
        visited = set() # stack
        for i in kids:
            if i not in visited:
                helper(i, visited)
        return visited

    def helper(id, visited):
        visited.add(id)
        visit_url = "https://hacker-news.firebaseio.com/v0/item/{}.json?print=pretty".format(id)
        visit_kid = json.loads(requests.get(visit_url).text)
        out.append(visit_kid)  #store output in outer source, every visit save node

        if 'kids' in visit_kid.keys():
            id_list = visit_kid['kids']
            for kid in id_list: # recursively visit kids of a node
                if kid not in visited:
                    helper(kid,visited)

    DSF()
    df = pd.DataFrame(out)
    df_filtered = df.loc[df['dead'] != True,]
    df_filtered['time'] = pd.to_datetime(df_filtered['time'], unit = 's')

    return df_filtered[['id', 'by', 'parent', 'text', 'time']]
    ...
