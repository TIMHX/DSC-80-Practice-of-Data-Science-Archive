# lab.py


import os
import pandas as pd
import numpy as np
import re


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def match_1(string):
    """
    >>> match_1("abcde]")
    False
    >>> match_1("ab[cde")
    False
    >>> match_1("a[cd]")
    False
    >>> match_1("ab[cd]")
    True
    >>> match_1("1ab[cd]")
    False
    >>> match_1("ab[cd]ef")
    True
    >>> match_1("1b[#d] _")
    True
    """
    # Your Code Here
    pattern = '^.{2}\[..\]'

    # Do not edit following code
    prog = re.compile(pattern)
    return prog.search(string) is not None


def match_2(string):
    """
    Phone numbers that start with '(858)' and
    follow the format '(xxx) xxx-xxxx' (x represents a digit)
    Notice: There is a space between (xxx) and xxx-xxxx
    >>> match_2("(123) 456-7890")
    False
    >>> match_2("858-456-7890")
    False
    >>> match_2("(858)45-7890")
    False
    >>> match_2("(858) 456-7890")
    True
    >>> match_2("(858)456-789")
    False
    >>> match_2("(858)456-7890")
    False
    >>> match_2("a(858) 456-7890")
    False
    >>> match_2("(858) 456-7890b")
    False
    """
    # Your Code Here
    pattern = '^\(858\)\s *\d{3}-\d{4}$'

    # Do not edit following code
    prog = re.compile(pattern)
    return prog.search(string) is not None


def match_3(string):
    """
    Find a pattern whose length is between 6 to 10
    and contains only word character, white space and ?.
    This string must have ? as its last character.
    >>> match_3("qwertsd?")
    True
    >>> match_3("qw?ertsd?")
    True
    >>> match_3("ab c?")
    False
    >>> match_3("ab   c ?")
    True
    >>> match_3(" asdfqwes ?")
    False
    >>> match_3(" adfqwes ?")
    True
    >>> match_3(" adf!qes ?")
    False
    >>> match_3(" adf!qe? ")
    False
    """
    # Your Code Here
    pattern = '^[a-zA-Z\s\?]{5,9}\?$'

    # Do not edit following code
    prog = re.compile(pattern)
    return prog.search(string) is not None


def match_4(string):
    """
    A string that begins with '$' and with another '$' within, where:
        - Characters between the two '$' can be anything except the 
        letters 'a', 'b', 'c' (lower case).
        - Characters after the second '$' can only have any number 
        of the letters 'a', 'b', 'c' (upper or lower case), with every 
        'a' before every 'b', and every 'b' before every 'c'.
            - E.g. 'AaBbbC' works, 'ACB' doesn't.
    >>> match_4("$$AaaaaBbbbc")
    True
    >>> match_4("$!@#$aABc")
    True
    >>> match_4("$a$aABc")
    False
    >>> match_4("$iiuABc")
    False
    >>> match_4("123$Abc")
    False
    >>> match_4("$$Abc")
    True
    >>> match_4("$qw345t$AAAc")
    False
    >>> match_4("$s$Bca")
    False
    """
    # Your Code Here
    pattern = '\$(([^abc].*)|())\$(?i)a*b+c+$'

    # Do not edit following code
    prog = re.compile(pattern)
    return prog.search(string) is not None


def match_5(string):
    """
    A string that represents a valid Python file name including the extension. 
    *Notice*: For simplicity, assume that the file name contains only letters, numbers and an underscore `_`.
    
    >>> match_5("dsc80.py")
    True
    >>> match_5("dsc80py")
    False
    >>> match_5("dsc80..py")
    False
    >>> match_5("dsc80+.py")
    False
    """
    # Your Code Here
    pattern = '(?i)^[a-z0-9_]+\.py$'

    # Do not edit following code
    prog = re.compile(pattern)
    return prog.search(string) is not None


def match_6(string):
    """
    Find patterns of lowercase letters joined with an underscore.
    >>> match_6("aab_cbb_bc")
    False
    >>> match_6("aab_cbbbc")
    True
    >>> match_6("aab_Abbbc")
    False
    >>> match_6("abcdef")
    False
    >>> match_6("ABCDEF_ABCD")
    False
    """

    # Your Code Here
    pattern = '^[a-z]+_[a-z]+$'

    # Do not edit following code
    prog = re.compile(pattern)
    return prog.search(string) is not None


def match_7(string):
    """
    Find patterns that start with and end with a _
    >>> match_7("_abc_")
    True
    >>> match_7("abd")
    False
    >>> match_7("bcd")
    False
    >>> match_7("_ncde")
    False
    """
    pattern = '^_.*_$'

    # Do not edit following code
    prog = re.compile(pattern)
    return prog.search(string) is not None


def match_8(string):
    """
    Apple registration numbers and Apple hardware product serial numbers 
    might have the number "0" (zero), but never the letter "O". 
    Serial numbers don't have the number "1" (one) or the letter "i". 
    
    Write a line of regex expression that checks 
    if the given Serial number belongs to a genuine Apple product.
    
    >>> match_8("ASJDKLFK10ASDO")
    False
    >>> match_8("ASJDKLFK0ASDo")
    True
    >>> match_8("JKLSDNM01IDKSL")
    False
    >>> match_8("ASDKJLdsi0SKLl")
    False
    >>> match_8("ASDJKL9380JKAL")
    True
    """
    pattern = '^[^O1i]+$'

    # Do not edit following code
    prog = re.compile(pattern)
    return prog.search(string) is not None


def match_9(string):
    '''
    >>> match_9('NY-32-NYC-1232')
    True
    >>> match_9('ca-23-SAN-1231')
    False
    >>> match_9('MA-36-BOS-5465')
    False
    >>> match_9('CA-56-LAX-7895')
    True
    '''
    pattern = '^(CA|NY)-\d{2}-(SAN|NYC|LAX)-\d{4}$'

    # Do not edit following code
    prog = re.compile(pattern)
    return prog.search(string) is not None


def match_10(string):
    '''
    Given an input string, cast it to lower case, remove spaces/punctuation, 
    and return a list of every 3-character substring that satisfy the following:
        - The first character doesn't start with 'a' or 'A'
        - The last substring (and only the last substring) can be shorter than 
        3 characters, depending on the length of the input string.
    
    >>> match_10('ABCdef')
    ['def']
    >>> match_10(' DEFaabc !g ')
    ['def', 'cg']
    >>> match_10('Come ti chiami?')
    ['com', 'eti', 'chi']
    >>> match_10('and')
    []
    >>> match_10( "Ab..DEF")
    ['def']
    
    '''
    out = string.lower()
    arr = pd.Series(re.findall(r'.{1,3}', out))
    arr_df = arr.str.extract(r'(^[^a].*)').dropna()
    arr = []
    for i in arr_df[0]:
        w = re.sub(r'[^\w]', '', i)
        w = re.sub(r'_', '', w)
        arr.append(w)

    arr = pd.Series(re.findall(r'.{1,3}', ''.join(arr)))
    if len(arr) == 0:
        return []
    return arr.str.extract(r'(^[^a].*)').dropna()[0].to_list()
    ...


# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def extract_personal(s):
    """
    :Example:
    >>> fp = os.path.join('data', 'messy.test.txt')
    >>> s = open(fp, encoding='utf8').read()
    >>> emails, ssn, bitcoin, addresses = extract_personal(s)
    >>> emails[0] == 'test@test.com'
    True
    >>> ssn[0] == '423-00-9575'
    True
    >>> bitcoin[0] == '1BvBMSEYstWetqTFn5Au4m4GFg7xJaNVN2'
    True
    >>> addresses[0] == '530 High Street'
    True
    """
    email = re.findall('[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+', s)
    ssn = re.findall('ssn\:([0-9]+-[0-9]+-[0-9]+)', s)
    bca = re.findall('bitcoin\:([0-9a-zA-Z]{5,})', s)
    sa = re.findall('([0-9]*\s[a-zA-Z\s]+)\\n\d*', s)
    return (email, ssn, bca, sa)
    ...


# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


def tfidf_data(review, reviews):
    """
    :Example:
    >>> fp = os.path.join('data', 'reviews.txt')
    >>> reviews = pd.read_csv(fp, header=None, squeeze=True)
    >>> review = open(os.path.join('data', 'review.txt'), encoding='utf8').read().strip()
    >>> out = tfidf_data(review, reviews)
    >>> out['cnt'].sum()
    85
    >>> 'before' in out.index
    True
    """
    wordlist = review.strip().split()
    index = set(wordlist)
    wordfreq = []
    for w in index:
        wordfreq.append(wordlist.count(w))
    key_bag = dict(zip(index, wordfreq))

    out_df = pd.DataFrame(pd.Series(key_bag), columns=['cnt'])
    tf = []
    for i in key_bag.keys():
        tf.append(review.count(i) / (review.count(' ') + 1))
    out_df['tf'] = tf

    idf = []
    for i in key_bag.keys():
        idf.append(np.log(len(reviews) / reviews.str.contains(i).sum()))
    out_df['idf'] = idf

    tfidf = out_df['tf'] * out_df['idf']
    out_df['tfidf'] = tfidf

    return out_df
    ...


def relevant_word(out):
    """
    :Example:
    >>> fp = os.path.join('data', 'reviews.txt')
    >>> reviews = pd.read_csv(fp, header=None, squeeze=True)
    >>> review = open(os.path.join('data', 'review.txt'), encoding='utf8').read().strip()
    >>> out = tfidf_data(review, reviews)
    >>> relevant_word(out) in out.index
    True
    """
    return out.tfidf.idxmax()
    ...


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


def hashtag_list(tweet_text):
    """
    :Example:
    >>> testdata = [['RT @DSC80: Text-cleaning is cool! #NLP https://t.co/xsfdw88d #NLP1 #NLP1']]
    >>> test = pd.DataFrame(testdata, columns=['text'])
    >>> out = hashtag_list(test['text'])
    >>> (out.iloc[0] == ['NLP', 'NLP1', 'NLP1'])
    True
    """
    tweet_text_c = tweet_text.copy()
    out = []
    for i in tweet_text_c.index:
        out.append(re.findall('\#([^\s]+)', tweet_text_c.loc[i]))
    return pd.Series(out)
    ...


def most_common_hashtag(tweet_lists):
    """
    :Example:
    >>> testdata = [['RT @DSC80: Text-cleaning is cool! #NLP https://t.co/xsfdw88d #NLP1 #NLP1']]
    >>> test = hashtag_list(pd.DataFrame(testdata, columns=['text'])['text'])
    >>> most_common_hashtag(test).iloc[0]
    'NLP1'
    """
    out = []
    frequancy =  pd.Series(tweet_lists.sum()).value_counts()
    for i in tweet_lists:
        d = {x: i.count(x) for x in i}
        if len(d) == 1:
            out.append(list(d.keys())[0])
        elif len(d) == 0:
            out.append(np.nan)
        else:
            a = {}
            for w in i:
                a[w] = frequancy[w]
            out.append(max(a, key=a.get))
    return pd.Series(out)
    ...


# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------


def create_features(ira):
    """
    :Example:
    >>> testdata = [['RT @DSC80: Text-cleaning is cool! #NLP https://t.co/xsfdw88d #NLP1 #NLP1']]
    >>> test = pd.DataFrame(testdata, columns=['text'])
    >>> out = create_features(test)
    >>> anscols = ['text', 'num_hashtags', 'mc_hashtags', 'num_tags', 'num_links', 'is_retweet']
    >>> ansdata = [['text cleaning is cool', 3, 'NLP1', 1, 1, True]]
    >>> ans = pd.DataFrame(ansdata, columns=anscols)
    >>> (out == ans).all().all()
    True
    """
    def clean_text(string):
        out = re.sub("RT.*\:\s", "", string)
        out = re.sub("http(s)://[^\s]+", "", out)
        out = re.sub("\#([^\s]+)", "", out)
        out = re.sub(r'\W+', ' ', out)
        return out.strip().lower()

    htag_list = hashtag_list(ira['text'])
    num_hashtags = htag_list.map(len)
    mc_hashtags = most_common_hashtag(htag_list)
    num_tags = []
    num_links = []
    is_retweet = []
    text = []
    for i in ira['text'].index:
        t = ira['text'].loc[i]
        num_tags.append(len(re.findall('\@[^\s]+', t)))
        num_links.append(len(re.findall('http(s)://[^\s]+', t)))
        is_retweet.append(True if re.match(r'^RT\s', t) else False)
        text.append(clean_text(t))
    out = pd.DataFrame()
    out['text'] = text
    out['num_hashtags'] = num_hashtags
    out['mc_hashtags'] = mc_hashtags
    out['num_tags'] = num_tags
    out['num_links'] = num_links
    out['is_retweet'] = is_retweet
    return out
    ...
