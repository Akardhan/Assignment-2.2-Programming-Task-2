import unittest
import numpy as np

from numcompute_stream.preprocessing import StandardScaler
from numcompute_stream.tree import DecisionTreeClassifier
from numcompute_stream.pipeline import Pipeline
from numcompute_stream.stream import StreamTrainer


class TestStreamTrainer(unittest.TestCase):

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

    def test_stream_trainer_fit_chunk(self):
        pipe = Pipeline([
            ("scale", StandardScaler()),
            ("model", DecisionTreeClassifier(max_depth=3)),
        ])

        trainer = StreamTrainer(pipe)
        trainer.fit_chunk(self.X[:4], self.y[:4])
        self.assertGreaterEqual(len(trainer.logs_), 1)

    def test_stream_trainer_score_chunk(self):
        pipe = Pipeline([
            ("scale", StandardScaler()),
            ("model", DecisionTreeClassifier(max_depth=3)),
        ])

        trainer = StreamTrainer(pipe)
        trainer.fit_chunk(self.X[:4], self.y[:4])
        score = trainer.score_chunk(self.X[4:], self.y[4:])
        self.assertTrue(0.0 <= score <= 1.0)

    def test_stream_trainer_multiple_chunks(self):
        pipe = Pipeline([
            ("scale", StandardScaler()),
            ("model", DecisionTreeClassifier(max_depth=3)),
        ])

        trainer = StreamTrainer(pipe)

        for i in range(0, len(self.X), 2):
            trainer.fit_chunk(self.X[i:i + 2], self.y[i:i + 2])

        self.assertGreaterEqual(len(trainer.logs_), 4)


if __name__ == "__main__":
    unittest.main()