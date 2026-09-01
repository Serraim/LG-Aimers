import unittest

import numpy as np
import pandas as pd

from scripts.s26_script import joint_success_probability, multilabel_success_probability


class FakeMultiClass:
    classes_ = np.arange(6)

    def predict_proba(self, matrix):
        values = np.zeros((len(matrix), 6), dtype="float64")
        values[:, 1] = 0.2
        values[:, 3] = 0.3
        values[:, 5] = 0.1
        values[:, 0] = 0.4
        return values


class FakeMultiLabel:
    def predict_proba(self, matrix):
        values = np.zeros((len(matrix), 5), dtype="float64")
        values[:, 0] = 0.61
        return values


class S26ScriptTest(unittest.TestCase):
    def test_multiclass_success_states_are_summed(self):
        matrix = pd.DataFrame({"x": [1, 2]})
        actual = joint_success_probability(FakeMultiClass(), matrix, [1, 3, 5])
        np.testing.assert_allclose(actual, [0.6, 0.6])

    def test_multilabel_uses_primary_success_head(self):
        matrix = pd.DataFrame({"x": [1, 2]})
        actual = multilabel_success_probability(FakeMultiLabel(), matrix)
        np.testing.assert_allclose(actual, [0.61, 0.61])


if __name__ == "__main__":
    unittest.main()
