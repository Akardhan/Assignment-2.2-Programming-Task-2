import unittest
import numpy as np

from numcompute_stream.preprocessing import StandardScaler, SimpleImputer, OneHotEncoder
from numcompute_stream.stats import StreamingStats
from numcompute_stream.metrics import StreamingAccuracy, StreamingConfusionMatrix
from numcompute_stream.tree import DecisionTreeClassifier
from numcompute_stream.ensemble import EnsembleClassifier


class TestEdgeCases(unittest.TestCase):

    def test_scaler_all_nan_column(self):
        X = np.array([[1, np.nan], [2, np.nan]], dtype=float)
        scaler = StandardScaler()
        Xt = scaler.fit_transform(X)
        self.assertFalse(np.isnan(Xt).any())

    def test_imputer_all_nan_column(self):
        X = np.array([[np.nan], [np.nan]], dtype=float)
        imputer = SimpleImputer(strategy="mean", fill_value=0.0)
        Xt = imputer.fit_transform(X)
        self.assertTrue(np.allclose(Xt, 0.0))

    def test_one_hot_unknown_category_ignored(self):
        X = np.array([["a"], ["b"]], dtype=object)
        encoder = OneHotEncoder()
        encoder.fit(X)
        Xt = encoder.transform(np.array([["c"]], dtype=object))
        self.assertEqual(np.sum(Xt), 0.0)

    def test_stats_empty_before_update(self):
        stats = StreamingStats()
        with self.assertRaises(RuntimeError):
            stats.mean()

    def test_accuracy_empty_result(self):
        metric = StreamingAccuracy()
        self.assertEqual(metric.result(), 0.0)

    def test_confusion_matrix_counts(self):
        metric = StreamingConfusionMatrix(classes=np.array([0, 1]))
        metric.update(np.array([0, 1]), np.array([1, 1]))
        cm = metric.result()
        self.assertEqual(np.sum(cm), 2)

    def test_tree_single_class(self):
        X = np.array([[1], [2], [3]], dtype=float)
        y = np.array([1, 1, 1])
        model = DecisionTreeClassifier(max_depth=2)
        model.fit(X, y)
        preds = model.predict(X)
        self.assertTrue(np.all(preds == 1))

    def test_ensemble_small_dataset(self):
        X = np.array([[0], [1]], dtype=float)
        y = np.array([0, 1])
        model = EnsembleClassifier(n_estimators=2, max_depth=1, random_state=1)
        model.fit(X, y)
        preds = model.predict(X)
        self.assertEqual(preds.shape, y.shape)


if __name__ == "__main__":
    unittest.main()