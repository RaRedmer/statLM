import numpy as np
from collections import Counter

if __name__ == '__main__':
    from ngram import NGramFrequenzy
else:
    from .ngram import NGramFrequenzy

class BaseStatisticalModel(object):
    def __init__(self, n_max=2, frequencies=None):
        self.n_max = n_max
        self.model_frequencies = frequencies
    
    def fit(self, corpus):
        """
        docstring
        """
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
        BaseStatisticalModel(frequencies={
            n: self.model_frequencies.get(n, NGramFrequenzy() ) + other.model_frequencies.get(n, NGramFrequenzy() )
            for n in range(1, common_max_n + 1)
        })


             

class NGramModel(BaseStatisticalModel):
    def __init__(self, n_max, **kwargs):
        super(NGramModel, self).__init__(n_max, **kwargs)

    def _recursive_search(self, parsed_query, n, top_n=1, **kwargs):
        search_result = self.model_frequencies[ n ].search_ngrams(query=parsed_query, **kwargs).most_common(top_n)
        if search_result:
            return search_result[0]
        else:
            n -= 1
            if n > 0:
                # enter recursion with first word removed from query
                return self._recursive_search( parsed_query[ 1: ], n )
            else:
                return np.NaN
  
    def predict(self, queries):
        """
        docstring
        """
        predictions = []
        for query in queries:
            parsed_query = query.split(" ")
            query_len = len( parsed_query )
            # adjust query if too long
            if query_len > self.n_max - 1:
                adjusted_query = parsed_query[ self.n_max + 1: ]
                # start recursive search
                pred = self._recursive_search(adjusted_query, n=self.n_max)
            else:
                # start recursive search
                pred = self._recursive_search(adjusted_query, n=query_len + 1)
            if isinstance(pred, tuple):
                predictions.append( pred[0].split(" ")[-1] )
            else:
                predictions.append( pred )
        return predictions

    def predict_proba(self, queries, top_n=1):
        """
        docstring
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


if __name__ == "__main__":
    test_corpus = [
        "let us see were this project leads us",
        "we are having great fun so far",
        "we are actively developing",
        "it is getting tougher but it is still fun",
        "this project teaches us how to construct test cases",
    ]   
    infer_doc = ["let us see were that project"]
    sb = NGramModel(n_max=3)
    # sb = BaseStatisticalModel(n_max=3)
    sb.fit( test_corpus )
    print(sb)
    print(sb.predict(infer_doc))
    print(sb.predict_proba(infer_doc))