import numpy as np
from collections import Counter

from .ngram import NGramFrequenzy


class BaseStatisticalModel(object):
    """Base class for all statistical models.
    """

    def __init__(self, n_max=2, frequencies=None):
        self.n_max = n_max
        self.model_frequencies = frequencies
    
    def _generate_freq(self, corpus):
        self.model_frequencies = { 
            n: NGramFrequenzy(corpus=corpus, ngram_range=( n, n ) ) 
            for n in range(1, self.n_max + 1)
        }

    @classmethod
    def __get_classname(cls):
        return cls.__name__

    def __repr__(self):
        default = f"{ self.__get_classname() }: n_max = { self.n_max }"
        if self.model_frequencies:
            ngram_stats = "\n".join([
                f"  {n}-grams: count={ len( freq.keys() ) }, freq={ sum( freq.values() ) }" 
                 for n, freq in self.model_frequencies.items()
            ])
            return f"{default}\n{ngram_stats} "
        else:
            return f"{default}, not fitted"

    def __add__(self, other):
        left_max_n = len(self.model_frequencies.keys())
        right_max_n = len(other.model_frequencies.keys())
        common_max_n = max( left_max_n, right_max_n )
        return BaseStatisticalModel(frequencies={
            n: self.model_frequencies.get(n, NGramFrequenzy() ) + other.model_frequencies.get(n, NGramFrequenzy() )
            for n in range(1, common_max_n + 1)
        })



class RecursiveNextWord(BaseStatisticalModel):
    """ Model which determines next word of a sequence by finding N-Gram with most matching words.

        Args:
            n_max (int): Maximum n-gram degree
    """
    def __init__(self, n_max, **kwargs):
        super(RecursiveNextWord, self).__init__(n_max, **kwargs)

    def _recursive_search(self, parsed_query, n, top_n=1, **kwargs):
        search_result = self.model_frequencies[ n ].search_ngrams(query=parsed_query, **kwargs).most_common(top_n)
        if search_result:
            return search_result[0]
        else:
            n -= 1
            if n > 1:
                # enter recursion with first word removed from query
                return self._recursive_search( parsed_query[ 1: ], n )
            else:
                return np.NaN
  
    def fit(self, corpus):
        """ Fit model by generating word frequencies from input corpus

            Args:
                corpus (iterable): Corpus as a sequence of docs each of type string
            
            Returns: 
                self
        """
        self._generate_freq(corpus)
        return self

    def predict(self, queries):
        """ Model predictions based on input queries

            Args:
                queries (iterable): sequence queries each of type string

            Returns:
                list: prediction for each query as list of words each of type string
        """
        predictions = []
        for query in queries:
            parsed_query = query.split(" ")
            query_len = len( parsed_query )
            # adjust query if too long
            adjusted_query = parsed_query[ -(self.n_max -1): ] if query_len > self.n_max - 1 else parsed_query
            pred = self._recursive_search(
                adjusted_query, 
                n=self.n_max, 
                top_n=1, 
                # normalize=True,
            )
            if isinstance(pred, tuple):
                # take last word ngram
                pred = pred[0].split(" ")[-1]
            predictions.append( pred )
        return predictions

    def predict_proba(self, queries, top_n=1):
        """ Model predictions and its conditional probability based on input queries

            Args:
                queries (iterable): sequence queries each of type string

            Returns:
                list: prediction for each query as list of words each of type string
        """
        probas = []
        for query in queries:
            parsed_query = query.split(" ")
            query_len = len( parsed_query )
            # adjust query if to long
            adjusted_query = parsed_query[ self.n_max + 1: ] if query_len > self.n_max - 1 else parsed_query
            pred = self._recursive_search(
                adjusted_query, 
                n=self.n_max, 
                top_n=1, 
                normalize=True,
            )
            probas.append( pred )
        return probas


# TODO: 
#     - additional language models (markov chain, backoff etc)
#     - methods for exploring word freq
#     - make ngram comparison more efficient by comparing lists instead of strings



if __name__ == "__main__":
    from ngram import NGramFrequenzy
    
    test_corpus = [
        "let us see were this project leads us",
        "we are having great fun so far",
        "we are actively developing",
        "it is getting tougher but it is still fun",
        "this project teaches us how to construct test cases",
    ]   
    infer_doc = ["let us see were that project",
                "we are",
                "it is",
                "it should be"]
    sb = RecursiveNextWord(n_max=3)
    # sb = BaseStatisticalModel(n_max=3)
    sb._generate_freq( test_corpus )
    print(sb)
    print(sb.predict(infer_doc))
    print(sb.predict_proba(infer_doc))