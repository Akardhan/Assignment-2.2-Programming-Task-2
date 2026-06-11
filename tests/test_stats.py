import unittest
import numpy as np

from numcompute_stream.stats import StreamingStats, nanmean, nanvariance


class TestStats(unittest.TestCase):

    def test_nanmean(self):
        X = np.array([1, 2, np.nan, 4], dtype=float)
        self.assertAlmostEqual(nanmean(X, axis=None), np.nanmean(X))

    def test_nanvariance(self):
        X = np.array([1, 2, np.nan, 4], dtype=float)
        self.assertAlmostEqual(nanvariance(X, axis=None), np.nanvar(X))

    def test_streaming_stats_mean(self):
        stats = StreamingStats()
        stats.update_stats(np.array([[1, 2], [3, 4]], dtype=float))
        self.assertTrue(np.allclose(stats.mean(), np.array([2, 3])))

    def test_streaming_stats_partial_updates(self):
        stats = StreamingStats()
        stats.update_stats(np.array([[1, 2]], dtype=float))
        stats.update_stats(np.array([[3, 4]], dtype=float))
        self.assertTrue(np.allclose(stats.mean(), np.array([2, 3])))

    def test_streaming_stats_variance(self):
        stats = StreamingStats()
        stats.update_stats(np.array([[1], [2], [3]], dtype=float))
        self.assertTrue(stats.variance()[0] >= 0)

    def test_streaming_stats_nan_handling(self):
        stats = StreamingStats()
        stats.update_stats(np.array([[1, np.nan], [3, 4]], dtype=float))
        self.assertFalse(np.isnan(stats.mean()[0]))

    def test_streaming_stats_quantile(self):
        stats = StreamingStats()
        stats.update_stats(np.array([[1], [2], [3], [4]], dtype=float))
        q = stats.quantile(0.5)
        self.assertTrue(np.isfinite(q).all())

    def test_streaming_stats_histogram(self):
        stats = StreamingStats()
        stats.update_stats(np.array([[1], [2], [3], [4]], dtype=float))
        counts, edges = stats.histogram(feature_index=0, bins=2)
        self.assertEqual(np.sum(counts), 4)
        self.assertEqual(len(edges), 3)

    def test_streaming_stats_reset(self):
        stats = StreamingStats()
        stats.update_stats(np.array([[1, 2]], dtype=float))
        stats.reset()
        with self.assertRaises(RuntimeError):
            stats.mean()


if __name__ == "__main__":
    unittest.main()