"""Stream training manager."""

from __future__ import annotations

import sys
from typing import Iterable, Tuple

import numpy as np

from .metrics import ClassificationMetrics, StreamingAccuracy


class StreamTrainer:
    """Manage chunk-wise training, evaluation and logging.

    ``model`` can be a bare estimator or a ``Pipeline``. Each call to
    ``fit_chunk`` trains on one chunk, predicts the same chunk for progressive
    feedback, and appends a history row with metrics and memory footprint.
    """

    def __init__(self, model, metrics: ClassificationMetrics | None = None) -> None:
        self.model = model
        self.metrics = metrics if metrics is not None else ClassificationMetrics()
        self.history_: list[dict] = []
        self.logs_ = self.history_
        self.chunk_index_: int = 0
        self.cumulative_accuracy_ = StreamingAccuracy()

    def fit_chunk(self, X, y) -> dict:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).ravel()
        if hasattr(self.model, 'partial_fit'):
            self.model.partial_fit(X, y)
        else:
            self.model.fit(X, y)
        y_pred = self.model.predict(X)
        self.metrics.update(y, y_pred)
        self.cumulative_accuracy_.update(y, y_pred)
        row = self._make_log_row(X, y, y_pred, phase='fit')
        self.history_.append(row)
        self.chunk_index_ += 1
        return row

    def score_chunk(self, X, y) -> float:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).ravel()
        y_pred = self.model.predict(X)
        self.metrics.update(y, y_pred)
        self.cumulative_accuracy_.update(y, y_pred)
        row = self._make_log_row(X, y, y_pred, phase='score')
        self.history_.append(row)
        self.chunk_index_ += 1
        return row['chunk_accuracy']

    def fit_stream(self, chunks: Iterable[Tuple[np.ndarray, np.ndarray]]) -> list[dict]:
        for X_chunk, y_chunk in chunks:
            self.fit_chunk(X_chunk, y_chunk)
        return self.history_

    def _make_log_row(self, X, y, y_pred, *, phase: str) -> dict:
        chunk_accuracy = float(np.mean(y == y_pred)) if y.shape[0] else 0.0
        metrics_result = self.metrics.result()
        memory_bytes = int(X.nbytes + y.nbytes + y_pred.nbytes + sys.getsizeof(self.model))
        return {
            'chunk': self.chunk_index_,
            'phase': phase,
            'n_samples': int(X.shape[0]),
            'chunk_accuracy': chunk_accuracy,
            'cumulative_accuracy': self.cumulative_accuracy_.result(),
            'rolling_accuracy': float(metrics_result.get('rolling_accuracy', 0.0)),
            'memory_bytes': memory_bytes,
        }

    def metric_values(self, key: str) -> list[float]:
        return [row[key] for row in self.history_ if key in row]

    def reset_history(self) -> None:
        self.history_.clear()
        self.chunk_index_ = 0
        self.cumulative_accuracy_.reset()
        self.metrics.reset()
