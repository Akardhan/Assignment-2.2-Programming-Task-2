import unittest
import numpy as np

from numcompute_stream.metrics import (
    StreamingAccuracy,
    StreamingPrecision,
    StreamingRecall,
    StreamingF1,
    StreamingConfusionMatrix,
    StreamingAUC,
    RollingConfusionMatrix,
)


class TestMetrics(unittest.TestCase):

    def test_accuracy(self):
        metric = StreamingAccuracy()
        metric.update(np.array([0, 1, 1]), np.array([0, 1, 0]))
        self.assertAlmostEqual(metric.result(), 2 / 3)

    def test_accuracy_reset(self):
        metric = StreamingAccuracy()
        metric.update(np.array([1]), np.array([1]))
        metric.reset()
        self.assertEqual(metric.result(), 0.0)

    def test_precision(self):
        metric = StreamingPrecision()
        metric.update(np.array([0, 1, 1, 0]), np.array([0, 1, 1, 1]))
        self.assertTrue(0.0 <= metric.result() <= 1.0)

    def test_recall(self):
        metric = StreamingRecall()
        metric.update(np.array([0, 1, 1, 0]), np.array([0, 1, 0, 0]))
        self.assertTrue(0.0 <= metric.result() <= 1.0)

    def test_f1(self):
        metric = StreamingF1()
        metric.update(np.array([0, 1, 1, 0]), np.array([0, 1, 0, 0]))
        self.assertTrue(0.0 <= metric.result() <= 1.0)

    def test_confusion_matrix(self):
        metric = StreamingConfusionMatrix(classes=np.array([0, 1]))
        metric.update(np.array([0, 1, 1, 0]), np.array([0, 1, 0, 0]))
        cm = metric.result()
        self.assertEqual(cm.shape, (2, 2))
        self.assertEqual(np.sum(cm), 4)

    def test_confusion_matrix_expands_mixed_labels(self):
        metric = StreamingConfusionMatrix(classes=np.array([0]))
        metric.update(np.array([0, "new"], dtype=object), np.array(["new", 0], dtype=object))
        self.assertEqual(metric.result().shape, (2, 2))

    def test_rolling_confusion_matrix_window(self):
        metric = RollingConfusionMatrix(classes=np.array([0, 1]), window_size=2)
        metric.update(np.array([0, 1, 1]), np.array([0, 1, 0]))
        self.assertEqual(np.sum(metric.result()), 2)

    def test_auc(self):
        metric = StreamingAUC()
        metric.update(np.array([0, 0, 1, 1]), np.array([0.1, 0.4, 0.6, 0.9]))
        score = metric.result()
        self.assertTrue(0.0 <= score <= 1.0)

    def test_metrics_multiple_updates(self):
        metric = StreamingAccuracy()
        metric.update(np.array([1, 0]), np.array([1, 1]))
        metric.update(np.array([0, 1]), np.array([0, 1]))
        self.assertAlmostEqual(metric.result(), 3 / 4)


if __name__ == "__main__":
    unittest.main()
