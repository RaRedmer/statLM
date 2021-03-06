******
statLM
******

.. image:: https://readthedocs.org/projects/statlm/badge/?version=latest
    :target: https://statlm.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status



statLM (Statistical Language Models) is a library for classical as well as modern language models.


Installation
############

::

    pip install statLM

Models Implemented
##################

* Stupid Backoff
* Naive N-Gram Model

Example Usage
#############

Train a language model and make predictions based on queries i.e. test data.

.. code-block:: python

    corpus = ["let us see were this project leads us",
                "we are having great fun so far",
                "we are actively developing",
                "it is getting tougher but it is still fun",
                "this project teaches us how to construct test cases"] 

    sb = StupidBackoff(n_max=3, alpha=0.4)
    # fit model on corpus
    sb.fit( corpus )
    # make predictions
    queries = ["let us see were that project", "how many options"]
    sb.predict(queries)


In Progress
###########

* additional language models
* improve efficiency of ngram comparisons
* construct CD/CI tests via github action
* add type checking


Contributing
############

**Setup**

1. Install `Poetry <https://python-poetry.org/>`__
2. Run ``make setup`` to prepare workspace from Makefile

**Testing**

1. Run ``make test`` to run all tests


Copyright
#########

Copyright (C) 2020 statLM Raphael Redmer

For license information, see LICENSE.txt.
