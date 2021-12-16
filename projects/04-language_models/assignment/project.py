# project.py


import pandas as pd
import numpy as np
import os
import re
import requests
import time


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def get_book(url):
    """
    get_book that takes in the url of a 'Plain Text UTF-8' book and 
    returns a string containing the contents of the book.

    The function should satisfy the following conditions:
        - The contents of the book consist of everything between 
        Project Gutenberg's START and END comments.
        - The contents will include title/author/table of contents.
        - You should also transform any Windows new-lines (\r\n) with 
        standard new-lines (\n).
        - If the function is called twice in succession, it should not 
        violate the robots.txt policy.

    :Example: (note '\n' don't need to be escaped in notebooks!)
    >>> url = 'http://www.gutenberg.org/files/57988/57988-0.txt'
    >>> book_string = get_book(url)
    >>> book_string[:20] == '\\n\\n\\n\\n\\nProduced by Chu'
    True
    """
    string = requests.get(url).text
    idx1 = re.search(r'\*\*\* START.*\*\*\*', string).end()
    idx2 = re.search(r'\*\*\* END.*\*\*\*', string).start()
    string_c = string[idx1:idx2]
    string_c = "\n".join(string_c.splitlines())
    return string_c
    ...


# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def tokenize(book_string):
    """
    tokenize takes in book_string and outputs a list of tokens 
    satisfying the following conditions:
        - The start of any paragraph should be represented in the 
        list with the single character \x02 (standing for START).
        - The end of any paragraph should be represented in the list 
        with the single character \x03 (standing for STOP).
        - Tokens in the sequence of words are split 
        apart at 'word boundaries' (see the regex lecture).
        - Tokens should include no whitespace.

    :Example:
    >>> test_fp = os.path.join('data', 'test.txt')
    >>> test = open(test_fp, encoding='utf-8').read()
    >>> tokens = tokenize(test)
    >>> tokens[0] == '\x02'
    True
    >>> tokens[9] == 'dead'
    True
    >>> sum([x == '\x03' for x in tokens]) == 4
    True
    >>> '(' in tokens
    True
    """

    def set_end(s):
        if s == 'EndParagraph':
            return '\x03'
        elif s == 'StartParagraph':
            return '\x02'
        elif s == ' ':
            del s
        elif s == '':
            del s
        else:
            return s.strip()

    string = re.sub('(\s){2,}', " EndParagraph StartParagraph ", book_string)
    string = re.sub('\\n', " ", string)
    string_list = re.split(r'\b', string)
    string_list = [i for i in string_list if ((i != ' ') and (i != ''))]
    string_list = map(set_end, string_list)
    string_list = list(string_list)

    if string_list[0] == '\x03':
        del string_list[0]
    if string_list[0] != '\x02':
        string_list.insert(0, '\x02')

    if string_list[-1] == '\x02':
        del string_list[0]
    if string_list[0] != '\x03':
        string_list.append('\x03')
    return string_list
    ...


# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


class UniformLM(object):
    """
    Uniform Language Model class.
    """

    def __init__(self, tokens):
        """
        Initializes a Uniform languange model using a
        list of tokens. It trains the language model
        using `train` and saves it to an attribute
        self.mdl.
        """
        self.mdl = self.train(tokens)

    def train(self, tokens):
        """
        Trains a uniform language model given a list of tokens.
        The output is a series indexed on distinct tokens, and
        values giving the (uniform) probability of a token occuring
        in the language.

        :Example:
        >>> tokens = tuple('one one two three one two four'.split())
        >>> unif = UniformLM(tokens)
        >>> isinstance(unif.mdl, pd.Series)
        True
        >>> set(unif.mdl.index) == set('one two three four'.split())
        True
        >>> (unif.mdl == 0.25).all()
        True
        """
        out = pd.Series(data=1 / len(set(tokens)), index=set(tokens))
        return out
        ...

    def probability(self, words):
        """
        probability gives the probabiliy a sequence of words
        appears under the language model.
        :param: words: a tuple of tokens
        :returns: the probability `words` appears under the language
        model.

        :Example:
        >>> tokens = tuple('one one two three one two four'.split())
        >>> unif = UniformLM(tokens)
        >>> unif.probability(('five',))
        0
        >>> unif.probability(('one', 'two')) == 0.0625
        True
        """
        out = 1
        for w in words:
            if w in self.mdl.index:
                out *= self.mdl.loc[w,]
            else:
                return 0
        return out
        ...

    def sample(self, M):
        """
        sample selects tokens from the language model of length M, returning
        a string of tokens.

        :Example:
        >>> tokens = tuple('one one two three one two four'.split())
        >>> unif = UniformLM(tokens)
        >>> samp = unif.sample(1000)
        >>> isinstance(samp, str)
        True
        >>> len(samp.split()) == 1000
        True
        >>> s = pd.Series(samp.split()).value_counts(normalize=True)
        >>> np.isclose(s, 0.25, atol=0.05).all()
        True
        """
        out = np.random.choice(self.mdl.index, M)
        return ' '.join(out)
        ...


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


class UnigramLM(object):

    def __init__(self, tokens):
        """
        Initializes a Unigram languange model using a
        list of tokens. It trains the language model
        using `train` and saves it to an attribute
        self.mdl.
        """
        self.mdl = self.train(tokens)
        self.ngrams = tokens

    def train(self, tokens):
        """
        Trains a unigram language model given a list of tokens.
        The output is a series indexed on distinct tokens, and
        values giving the probability of a token occuring
        in the language.

        :Example:
        >>> tokens = tuple('one one two three one two four'.split())
        >>> unig = UnigramLM(tokens)
        >>> isinstance(unig.mdl, pd.Series)
        True
        >>> set(unig.mdl.index) == set('one two three four'.split())
        True
        >>> unig.mdl.loc['one'] == 3 / 7
        True
        """
        unique_w = dict.fromkeys(set(tokens), 0)
        for w in unique_w:
            w_count = tokens.count(w)
            unique_w[w] = w_count / len(tokens)
        return pd.Series(unique_w)
        ...

    def probability(self, words):
        """
        probability gives the probabiliy a sequence of words
        appears under the language model.
        :param: words: a tuple of tokens
        :returns: the probability `words` appears under the language
        model.

        :Example:
        >>> tokens = tuple('one one two three one two four'.split())
        >>> unig = UnigramLM(tokens)
        >>> unig.probability(('five',))
        0
        >>> p = unig.probability(('one', 'two'))
        >>> np.isclose(p, 0.12244897959, atol=0.0001)
        True
        """
        out = 1
        for w in words:
            if w in self.mdl.index:
                out *= self.mdl.loc[w,]
            else:
                return 0
        return out
        ...
        ...

    def sample(self, M):
        """
        sample selects tokens from the language model of length M, returning
        a string of tokens.

        >>> tokens = tuple('one one two three one two four'.split())
        >>> unig = UnigramLM(tokens)
        >>> samp = unig.sample(1000)
        >>> isinstance(samp, str)
        True
        >>> len(samp.split()) == 1000
        True
        >>> s = pd.Series(samp.split()).value_counts(normalize=True).loc['one']
        >>> np.isclose(s, 0.41, atol=0.05).all()
        True
        """
        out = np.random.choice(self.mdl.index, M, p=self.mdl)
        return ' '.join(out)
        ...


# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------


class NGramLM(object):

    def __init__(self, N, tokens):
        """
        Initializes a N-gram languange model using a
        list of tokens. It trains the language model
        using `train` and saves it to an attribute
        self.mdl.
        """

        self.N = N
        self.tokens = tokens

        ngrams = self.create_ngrams(tokens)
        self.ngrams = ngrams
        self.mdl = self.train(ngrams)

        if N < 2:
            raise Exception('N must be greater than 1')
        elif N == 2:
            self.prev_mdl = UnigramLM(tokens)
        else:
            mdl = NGramLM(N - 1, tokens)
            self.prev_mdl = mdl

    def create_ngrams(self, tokens):
        """
        create_ngrams takes in a list of tokens and returns a list of N-grams. 
        The START/STOP tokens in the N-grams should be handled as 
        explained in the notebook.

        :Example:
        >>> tokens = tuple('\x02 one two three one four \x03'.split())
        >>> bigrams = NGramLM(2, [])
        >>> out = bigrams.create_ngrams(tokens)
        >>> isinstance(out[0], tuple)
        True
        >>> out[0]
        ('\\x02', 'one')
        >>> out[2]
        ('two', 'three')
        """
        # temp = zip(*[tokens[i:] for i in range(0, self.N)])
        # ans = [tuple(ngram) for ngram in temp]
        # return ans
        out = []
        for i in range(len(tokens)):
            temp = []
            j = self.N
            k = i
            while j > 0 and k < len(tokens):
                temp.append(tokens[k])
                k += 1
                j -= 1
            out.append(tuple(temp))
        out = out[:-(self.N - 1)]
        return tuple(out)
        ...

    def train(self, ngrams):
        """
        Trains a n-gram language model given a list of tokens.
        The output is a dataframe with three columns (ngram, n1gram, prob).

        :Example:
        >>> tokens = tuple('\x02 one two three one four \x03'.split())
        >>> bigrams = NGramLM(2, tokens)
        >>> set(bigrams.mdl.columns) == set('ngram n1gram prob'.split())
        True
        >>> bigrams.mdl.shape == (6, 3)
        True
        >>> bigrams.mdl['prob'].min() == 0.5
        True
        """
        out = pd.DataFrame()
        out['ngram'] = self.ngrams
        out['n1gram'] = out['ngram'].apply(lambda x: (x[:-1]))
        nCounts = out['ngram'].value_counts()
        n1Counts = out['n1gram'].value_counts()

        out['t'] = out['n1gram'].apply(lambda x: n1Counts[x])
        out['prob'] = out['ngram'].apply(lambda x: nCounts[x])
        out['prob'] = out['prob'] / out['t']
        out = out.drop(columns='t')
        return out

    def probability(self, words):
        """
        probability gives the probabiliy a sequence of words
        appears under the language model.
        :param: words: a tuple of tokens
        :returns: the probability `words` appears under the language
        model.

        :Example:
        >>> tokens = tuple('\x02 one two one three one two \x03'.split())
        >>> bigrams = NGramLM(2, tokens)
        >>> p = bigrams.probability('two one three'.split())
        >>> np.isclose(p, (1/4)*(1/2)*(1/3))
        True
        >>> bigrams.probability('one two five'.split()) == 0
        True
        """
        prob = 1
        fore_set = []
        fore_set.append(words[0])
        for i in range(1, self.N - 1):
            fore_set.insert(0, tuple(words[0:i + 1]))

        prev = self
        for i in fore_set:
            prev = prev.prev_mdl
            if type(i) == tuple:
                prob *= prev.mdl[prev.mdl['ngram'] == i]['prob']
            else:
                prob *= np.array(prev.mdl[i])

        ng = self.create_ngrams(words)

        for i in ng:
            p = np.array(self.mdl[self.mdl['ngram'] == i]['prob'])
            try:
                prob = np.array(p * prob)
            except:
                return 0
        return prob[0]
        ...

    def sample(self, M):
        """
        sample selects tokens from the language model of length M, returning
        a string of tokens.

        :Example:
        >>> tokens = tuple('\x02 one two three one four \x03'.split())
        >>> bigrams = NGramLM(2, tokens)
        >>> samp = bigrams.sample(3)
        >>> len(samp.split()) == 4  # don't count the initial START token.
        True
        >>> samp[:2] == '\\x02 '
        True
        >>> set(samp.split()) <= {'\\x02', '\\x03', 'one', 'two', 'three', 'four'}
        True
        """
        ...
        out = ["\x02"]
        if self.N != 2:
            self.helper(out)
        while len(out) <= M:
            prefix = out[-(self.N - 1):]
            try:
                valid = self.mdl[self.mdl.n1gram == tuple(prefix)]
                rand = np.random.choice(
                    valid['ngram'].values,
                    size=1,
                    p=valid['prob'].values * 1 / sum(valid['prob'].values)
                )[0][-1]
                out.append(rand)
            except:
                out.append("\x03")
        return ' '.join(out)

    def helper(self, out):
        prev = self.prev_mdl
        mdls = []
        while prev.N > 2:
            mdls.append(prev.mdl)
            prev = prev.prev_mdl
        mdls.append(prev.mdl)
        mdls = mdls[::-1]   # reverse
        for mdl in mdls:
            valid = mdl[mdl['n1gram'] == tuple(out)]
            rand = np.random.choice(
                valid['ngram'].values,
                size=1,
                p=valid['prob'].values * 1 / sum(valid['prob'].values)
            )[0][-1]
            out.append(rand)
        ...
