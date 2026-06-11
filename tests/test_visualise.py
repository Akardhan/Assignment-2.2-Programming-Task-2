import unittest
import tempfile
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")

from numcompute_stream.visualise import (
    plot_metric_over_time,
    compare_models,
    plot_predictions_vs_ground_truth,
)


class TestVisualise(unittest.TestCase):

    def test_plot_metric_over_time_saves_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metric.png"
            plot_metric_over_time(
                [0.5, 0.6, 0.7],
                title="Accuracy Over Time",
                ylabel="Accuracy",
                save_path=str(path),
                show=False,
            )
            self.assertTrue(path.exists())

    def test_compare_models_saves_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "compare.png"
            compare_models(
                [0.5, 0.6, 0.7],
                [0.4, 0.65, 0.72],
                labels=("Tree", "Forest"),
                save_path=str(path),
                show=False,
            )
            self.assertTrue(path.exists())

    def test_predictions_vs_ground_truth_saves_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.png"
            plot_predictions_vs_ground_truth(
                np.array([0, 1, 1, 0]),
                np.array([0, 1, 0, 0]),
                save_path=str(path),
                show=False,
            )
            self.assertTrue(path.exists())


if __name__ == "__main__":
    unittest.main()