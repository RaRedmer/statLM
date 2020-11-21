# -*- coding: utf-8 -*-

# from .context import statLM

# from statLM.statistical_models import RecursiveNextWord
import unittest
import numpy as np

from context import statistical_models as sm

class BasicTestSuite(unittest.TestCase):
    """Basic test cases."""

    def test_statistical_models(self):
        test_corpus = [
            "let us see were this project leads us",
            "we are having great fun so far",
            "we are actively developing",
            "it is getting tougher but it is still fun",
            "this project teaches us how to construct test cases",
        ]        
        sb = sm.RecursiveNextWord(n_max=3)
        sb.fit( test_corpus )
        infer_doc = ["let us see were that project",
                    "we are",
                    "it is",
                    "it should be"]
        self.assertEqual(
            sb.predict(infer_doc), 
            ['leads', 'actively', 'getting', np.NaN]
        )

# TODO: 
#     - construct more test cases
    # - automate test via github action

if __name__ == '__main__':
    unittest.main()