import unittest
import numpy as np

from numcompute_stream.preprocessing import StandardScaler, SimpleImputer
from numcompute_stream.tree import DecisionTreeClassifier
from numcompute_stream.pipeline import Pipeline


class TestPipeline(unittest.TestCase):

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

    def test_pipeline_fit_predict(self):
        pipe = Pipeline([
            ("impute", SimpleImputer()),
            ("scale", StandardScaler()),
            ("model", DecisionTreeClassifier(max_depth=3)),
        ])

        pipe.fit(self.X, self.y)
        preds = pipe.predict(self.X)
        self.assertEqual(preds.shape, self.y.shape)

    def test_pipeline_partial_fit(self):
        pipe = Pipeline([
            ("impute", SimpleImputer()),
            ("scale", StandardScaler()),
            ("model", DecisionTreeClassifier(max_depth=3)),
        ])

        pipe.partial_fit(self.X[:3], self.y[:3])
        pipe.partial_fit(self.X[3:], self.y[3:])
        preds = pipe.predict(self.X)
        self.assertEqual(preds.shape, self.y.shape)

    def test_pipeline_transform_returns_preprocessed_features(self):
        pipe = Pipeline([
            ("impute", SimpleImputer()),
            ("scale", StandardScaler()),
            ("model", DecisionTreeClassifier(max_depth=3)),
        ])

        pipe.fit(self.X, self.y)
        Xt = pipe.transform(self.X)
        self.assertEqual(Xt.shape, self.X.shape)
        self.assertTrue(np.all(np.isfinite(Xt)))

    def test_pipeline_predict_proba(self):
        pipe = Pipeline([
            ("scale", StandardScaler()),
            ("model", DecisionTreeClassifier(max_depth=3)),
        ])

        pipe.fit(self.X, self.y)
        proba = pipe.predict_proba(self.X)
        self.assertEqual(proba.shape[0], self.X.shape[0])

    def test_pipeline_invalid_steps(self):
        with self.assertRaises(ValueError):
            Pipeline([])


if __name__ == "__main__":
    unittest.main()
