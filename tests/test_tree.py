import unittest
import numpy as np

from numcompute_stream.tree import DecisionTreeClassifier


class TestDecisionTree(unittest.TestCase):

    def setUp(self):
        self.X = np.array([
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
            [2, 2],
            [2, 3],
        ], dtype=float)

        self.y = np.array([0, 0, 1, 1, 1, 1])

    def test_tree_fit_predict(self):
        model = DecisionTreeClassifier(max_depth=3)
        model.fit(self.X, self.y)
        preds = model.predict(self.X)
        self.assertEqual(preds.shape, self.y.shape)

    def test_tree_partial_fit(self):
        model = DecisionTreeClassifier(max_depth=3)
        model.partial_fit(self.X[:3], self.y[:3])
        model.partial_fit(self.X[3:], self.y[3:])
        preds = model.predict(self.X)
        self.assertEqual(preds.shape, self.y.shape)

    def test_tree_predict_proba(self):
        model = DecisionTreeClassifier(max_depth=3)
        model.fit(self.X, self.y)
        proba = model.predict_proba(self.X)
        self.assertEqual(proba.shape[0], self.X.shape[0])
        self.assertTrue(np.allclose(np.sum(proba, axis=1), 1.0))

    def test_tree_handles_nan(self):
        X = self.X.copy()
        X[0, 0] = np.nan
        model = DecisionTreeClassifier(max_depth=3)
        model.fit(X, self.y)
        preds = model.predict(X)
        self.assertEqual(preds.shape, self.y.shape)

    def test_tree_invalid_depth(self):
        with self.assertRaises(ValueError):
            DecisionTreeClassifier(max_depth=0)

    def test_tree_invalid_max_samples(self):
        with self.assertRaises(ValueError):
            DecisionTreeClassifier(max_samples=0)

    def test_tree_fit_resets_feature_count(self):
        model = DecisionTreeClassifier(max_depth=2)
        model.fit(self.X, self.y)
        model.fit(np.array([[0, 1, 2], [1, 2, 3]], dtype=float), np.array([0, 1]))
        self.assertEqual(model.n_features_in_, 3)

    def test_tree_summary(self):
        model = DecisionTreeClassifier(max_depth=3)
        model.fit(self.X, self.y)
        if hasattr(model, "summary"):
            summary = model.summary()
            self.assertIsInstance(summary, dict)


if __name__ == "__main__":
    unittest.main()
