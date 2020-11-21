******
statLM
******
statLM (Statistical Language Models) is a library for classical as well as modern language models.

Installation
############



Example Usage
#############

::

    corpus = ["let us see were this project leads us",
                    "we are having great fun so far",
                    "we are actively developing",
                    "it is getting tougher but it is still fun",
                    "this project teaches us how to construct test cases"] 

    rn = RecursiveNextWord(n_max=3)
    # fit model on corpus
    rn.fit( test_corpus )
    # make predictions
    queries = ["let us see were that project", "but it"]
    rn.predict(queries)


In Progress
###########

* More language models e.g. (Stupid) Backoff, N-Gram Model
* improve efficiency of ngram comparisons


Copyright
#########

Copyright (C) 2020 statLM Raphael Redmer

For license information, see LICENSE.txt.

