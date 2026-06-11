import unittest
import numpy as np

from numcompute_stream.ensemble import EnsembleClassifier, BoostingClassifier


class TestEnsemble(unittest.TestCase):

    def setUp(self):
        self.X = np.array([
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
            [2, 2],
            [2, 3],
            [3, 2],
            [3, 3],
        ], dtype=float)

        self.y = np.array([0, 0, 1, 1, 1, 1, 0, 0])

    def test_ensemble_fit_predict(self):
        model = EnsembleClassifier(n_estimators=3, max_depth=3, random_state=1)
        model.fit(self.X, self.y)
        preds = model.predict(self.X)
        self.assertEqual(preds.shape, self.y.shape)

    def test_ensemble_partial_fit(self):
        model = EnsembleClassifier(n_estimators=3, max_depth=3, random_state=1)
        model.partial_fit(self.X[:4], self.y[:4])
        model.partial_fit(self.X[4:], self.y[4:])
        preds = model.predict(self.X)
        self.assertEqual(preds.shape, self.y.shape)

    def test_ensemble_predict_proba(self):
        model = EnsembleClassifier(n_estimators=3, max_depth=3, random_state=1)
        model.fit(self.X, self.y)
        proba = model.predict_proba(self.X)
        self.assertEqual(proba.shape[0], self.X.shape[0])
        self.assertTrue(np.allclose(np.sum(proba, axis=1), 1.0))

    def test_ensemble_invalid_estimators(self):
        with self.assertRaises(ValueError):
            EnsembleClassifier(n_estimators=0)

    def test_ensemble_invalid_depth(self):
        with self.assertRaises(ValueError):
            EnsembleClassifier(max_depth=0)

    def test_ensemble_fit_resets_feature_count(self):
        model = EnsembleClassifier(n_estimators=3, max_depth=2, random_state=1)
        model.fit(self.X, self.y)
        X_new = np.array([[0, 1, 2], [1, 2, 3], [3, 4, 5]], dtype=float)
        y_new = np.array([0, 1, 1])
        model.fit(X_new, y_new)
        self.assertEqual(model.n_features_in_, 3)

    def test_boosting_fit_predict(self):
        model = BoostingClassifier(n_estimators=3, max_depth=1, random_state=1)
        model.fit(self.X, self.y)
        preds = model.predict(self.X)
        self.assertEqual(preds.shape, self.y.shape)

    def test_boosting_partial_fit(self):
        model = BoostingClassifier(n_estimators=3, max_depth=1, random_state=1)
        model.partial_fit(self.X[:4], self.y[:4])
        model.partial_fit(self.X[4:], self.y[4:])
        preds = model.predict(self.X)
        self.assertEqual(preds.shape, self.y.shape)

    def test_boosting_summary(self):
        model = BoostingClassifier(n_estimators=3, max_depth=1, random_state=1)
        model.fit(self.X, self.y)
        self.assertIn("n_estimators_used", model.summary())


if __name__ == "__main__":
    unittest.main()
