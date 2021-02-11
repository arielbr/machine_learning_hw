""" 
Keep kernel implementations in here.
"""

import numpy as np
from collections import defaultdict, Counter
from functools import wraps
from tqdm import tqdm
import math
import scipy


def cache_decorator():
    """
    Cache decorator. Stores elements to avoid repeated computations.
    For more details see: https://stackoverflow.com/questions/36684319/decorator-for-a-class-method-that-caches-return-value-after-first-access
    """
    def wrapper(function):
        """
        Return element if in cache. Otherwise compute and store.
        """
        cache = {}

        @wraps(function)
        def element(*args):
            if args in cache:
                result = cache[args]
            else:
                result = function(*args)
                cache[args] = result
            return result

        def clear():
            """
            Clear cache.
            """
            cache.clear()

        # Clear the cache
        element.clear = clear
        return element
    return wrapper


class Kernel(object):
    """ Abstract kernel object.
    """

    def evaluate(self, s, t):
        """
        Kernel function evaluation.

        Args:
            s: A string corresponding to a document.
            t: A string corresponding to a document.

        Returns:
            A float from evaluating K(s,t)
        """
        raise NotImplementedError()

    def compute_kernel_matrix(self, *, X, X_prime=None):
        """
        Compute kernel matrix. Index into kernel matrix to evaluate kernel function.

        Args:
            X: A list of strings, where each string corresponds to a document.
            X_prime: (during testing) A list of strings, where each string corresponds to a document.

        Returns:
            A compressed sparse row matrix of floats with each element representing
            one kernel function evaluation.
        """
        X_prime = X if not X_prime else X_prime
        kernel_matrix = np.zeros((len(X), len(X_prime)), dtype=np.float32)

        for i in range(len(X)):
            for j in range(len(X_prime)):
                kernel_matrix[i][j] = self.evaluate(X[i], X_prime[j])
        return kernel_matrix


class NgramKernel(Kernel):
    def __init__(self, *, ngram_length):
        """
        Args:
            ngram_length: length to use for n-grams
        """
        self.ngram_length = ngram_length

    def generate_ngrams(self, doc):
        """
        Generate the n-grams for a document.

        Args:
            doc: A string corresponding to a document.

        Returns:
            Set of all distinct n-grams within the document.
        """
        l = len(doc)
        # set of unique ngrams
        s = set()
        for i in range(l-self.ngram_length+1):
            s.add(doc[i:i+self.ngram_length])
        return s

    @cache_decorator()
    def evaluate(self, s, t):
        """
        n-gram kernel function evaluation.

        Args:
            s: A string corresponding to a document.
            t: A string corresponding to a document.

        Returns:
            A float from evaluating K(s,t)
        """
        s1 = self.generate_ngrams(s)
        s2 = self.generate_ngrams(t)
        intersection = s1.intersection(s2)
        union = s1.union(s2)
        if len(union) == 0:
            return 1
        else:
            return len(intersection) / len(union)


class TFIDFKernel(Kernel):
    def __init__(self, *, X, X_prime=None):
        """
        Pre-compute tf-idf values for each (document, word) pair in dataset.

        Args:
            X: A list of strings, where each string corresponds to a document.
            X_prime: (during testing) A list of strings, where each string corresponds to a document.

        Sets:
            tfidf: You will use this in the evaluate function.
        """
        self.tfidf = self.compute_tfidf(X, X_prime)

    def compute_tf(self, doc):
        """
        Compute the tf for each word in a particular document.
        You may choose to use or not use this helper function.

        Args:
            doc: A string corresponding to a document.

        Returns:
            A data structure containing tf values.
        """
        d = dict()
        # list of words, each can repeat multiple times
        l = doc.split(" ")
        # get total count of words
        for word in l:
            if word not in d.keys():
                d[word] = 1
            else:
                d[word] += 1
        # divide by the number of words in d
        for k in d.keys():
            d[k] /= len(l)
        return d

    def compute_df(self, X, vocab):
        """
        Compute the df for each word in the vocab.
        You may choose to use or not use this helper function.

        Args:
            X: A list of strings, where each string corresponds to a document.
            vocab: A set of distinct words that occur in the corpus.

        Returns:
            A data structure containing df values.
        """
        df = dict()
        for v in vocab:
            df[v] = 0
        # list of set, each contains all words in a doc
        doc_list = []
        for x in X:
            doc_list.append(set(x.split(" ")))
        for doc in doc_list:
            for word in df.keys():
                if word in doc:
                    df[word] += 1
        return df

    def compute_tfidf(self, X, X_prime):
        """
        Compute the tf-idf for each (document, word) pair in dataset.
        You will call the helper functions to compute term-frequency 
        and document-frequency here.

        Args:
            X: A list of strings, where each string corresponds to a document.
            X_prime: (during testing) A list of strings, where each string corresponds to a document.

        Returns:
            A data structure containing tf-idf values. You can represent this however you like.
            If you're having trouble, you may want to consider a dictionary keyed by 
            the tuple (document, word).
        """
        # Concatenate collections of documents during testing
        if X_prime:
            X = X + X_prime

        vocab = set()
        # generate the set of all vovab
        for x in X:
            words = set(x.split(" "))
            vocab = vocab.union(words)

        tfidf = dict()
        # returned: a dict of the df (count) of each word in vocab
        df = self.compute_df(X, vocab)
        idf = dict()
        for word in vocab:
            idf[word] = np.log(len(X) / (df[word] + 1))

        # tf-idf(w, d) = tf(w, d) · log(N / (df(w) + 1))
        for doc in X:
            # returned: a dict of tf value (freq) of all words words: tf
            tf = self.compute_tf(doc)
            for word in vocab:
                if word not in tf.keys():
                    tfidf[(word, doc)] = 0 * idf[word]
                else:
                    tfidf[(word, doc)] = tf[word] * idf[word]
        return tfidf

    @cache_decorator()
    def evaluate(self, s, t):
        """
        tf-idf kernel function evaluation.

        Args:
            s: A string corresponding to a document.
            t: A string corresponding to a document.

        Returns:
            A float from evaluating K(s,t)
        """
        s_set = set(s.split(" "))
        k = 0

        t_tf = self.compute_tf(t)
        for s_word in s_set:
            if s_word not in t_tf.keys():
                continue
            else:
                freq = t_tf[s_word]
                k += freq * self.tfidf[(s_word, s)]
        return k
