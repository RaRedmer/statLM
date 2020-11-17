# -*- coding: utf-8 -*-
# from . import helpers
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from collections.abc import Iterable


class NGramFrequenzy(object):
    """
    docstring
    """
    def __init__(self, corpus=[], frequency={}, **kwargs):
        if corpus:
            self.__ngram_freq = self.__extract_ngram_frequency(corpus=corpus, **kwargs)
        elif frequency:
            if isinstance(frequency, dict):
                self.__ngram_freq = Counter(frequency)
            elif isinstance(frequency, Counter):
                self.__ngram_freq = frequency
            else:
                raise ValueError("Submitted frequency has to be either dict or Counter object")
        else:
            self.__ngram_freq =  Counter()


    @staticmethod
    def __extract_ngram_frequency(corpus, **kwargs):
        """ Count occurences for each word in the whole corpus.
            Arg:
                corpus (pd.Series): Series of documents each formatted as string
                
            Returns:
                pd.Series: Counts indexed by its respective word
        """
        cv = CountVectorizer(**kwargs)
        # get bag-of-words as sparse matrix
        bow = cv.fit_transform( corpus )
        # obtain list of feature names
        vocab = list( cv.get_feature_names() )
        # # word occurence for each column and collapse to vector
        word_counts = bow.sum(axis=0).A1
        # map word to its counts
        freq_distribution = Counter( dict( zip( vocab, word_counts) ) )
        return freq_distribution

    @classmethod
    def from_corpus(cls, corpus, **kwargs):
        """
        docstring
        """
        if isinstance(corpus, list):
            return cls( frequency=cls.__extract_ngram_frequency(corpus, **kwargs) )
        else:
            raise  ValueError("Corpus has to be a list object")

    @classmethod
    def from_frequency(cls, frequency):
        """
        docstring
        """
        if isinstance(frequency, dict) or isinstance(frequency, Counter):
            return cls( frequency = frequency )
        else:
            raise ValueError("N-Gram Frequency has to be either dict or Counter object")

    def __repr__(self):
        if self.__ngram_freq:
            return str(self.__ngram_freq)

    def __str__(self):
        if self.__ngram_freq:
            return str( dict(self.__ngram_freq) )

    def __add__(self, other):
        return NGramFrequenzy(frequency=self.__ngram_freq + other.__ngram_freq)

    def __iter__(self):
        for ngram, count in self.__ngram_freq.items():
            yield ngram, count

    def __getitem__(self, val):
        if isinstance(val, str):
            return self.__ngram_freq[val]
        elif isinstance(val, list):
            return self.search_ngrams(query=val)
        else:
            raise ValueError("Subcription value has to be either str or list")

    def keys(self):
        """
        docstring
        """
        return self.__ngram_freq.keys()

    def values(self):
        """
        docstring
        """
        return self.__ngram_freq.values()        

    def search_ngrams(self, query, normalize=False):
        """
        docstring
        """
        parsed_query = query.split(" ") if isinstance(query, str) else query
        word_num = len(parsed_query)        
        query_result = { 
            ngram: count 
            for ngram, count in self.__ngram_freq.items() 
            if ngram.split(" ")[ :word_num ] == parsed_query
        }
        if not normalize:
            return NGramFrequenzy( frequency= query_result )
        else:
            total_freq = sum( query_result.values() )
            return NGramFrequenzy( frequency= { ngram: (count / total_freq) for ngram, count in query_result.items() } )

    def most_common(self, top_n=1):
        return self.__ngram_freq.most_common(top_n)
        # else:


# TODO:
# - try out to inherit from Counter

if __name__ == "__main__":
    test_corpus = [
        "let us see were this project leads us",
        "we are having great fun so far",
        "we are having even greater fun now",
        "we are actively developing",
        "it is getting tougher but it is still fun",
        "this project teaches us how to construct test cases",
        "this project teaches us how to construct a package",
        "this project teaches us how to construct a setup tool",
        "this project teaches us how to build an api documentations",
    ]   
    ngfreq = NGramFrequenzy(corpus=test_corpus, ngram_range=(3,3))

    ngfreq_text = NGramFrequenzy(corpus=test_corpus[:2], ngram_range=(3,3))
    # print(ngfreq_text)
    # print(ngfreq + ngfreq_text)
    print(ngfreq)
    # print(ngfreq["are having"])
    print( ngfreq.most_common(1) )
    print( ngfreq.search_ngrams(query="we are", normalize=True).most_common(3) )
    print( ngfreq[ ["we", "are"] ].most_common(3) )
    # print( dict(ngfreq.search_ngrams(query="we are", normalize=True)))
