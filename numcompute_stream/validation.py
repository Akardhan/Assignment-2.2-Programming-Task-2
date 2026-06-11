"""Progressive validation utilities for streaming experiments."""

from __future__ import annotations

from copy import deepcopy
from typing import Dict, Optional

import numpy as np

from .datasets import iter_chunks
from .metrics import ClassificationMetrics


class ProgressiveValidator:
    """Compare multiple streaming models over the same sequence of chunks.

    Each model is evaluated chunk by chunk. For the first chunk, the model is
    trained before scoring because an unfitted estimator cannot predict. For all
    later chunks, predictions are recorded first and then the model is updated,
    which approximates prequential evaluation.
    """

    def __init__(self, models: Dict[str, object], chunk_size: int = 50, clone_models: bool = True) -> None:
        if not models:
            raise ValueError('models must not be empty.')
        if chunk_size <= 0:
            raise ValueError('chunk_size must be positive.')
        self.models = deepcopy(models) if clone_models else models
        self.chunk_size = chunk_size
        self.metrics_: dict[str, ClassificationMetrics] = {name: ClassificationMetrics() for name in self.models}
        self.history_: dict[str, list[dict]] = {name: [] for name in self.models}

    def evaluate(self, X, y) -> dict[str, list[dict]]:
        for chunk_index, (X_chunk, y_chunk) in enumerate(iter_chunks(X, y, chunk_size=self.chunk_size)):
            for name, model in self.models.items():
                if chunk_index == 0:
                    self._fit(model, X_chunk, y_chunk)
                    pred = model.predict(X_chunk)
                else:
                    pred = model.predict(X_chunk)
                    self._fit(model, X_chunk, y_chunk)
                self.metrics_[name].update(y_chunk, pred)
                metric_values = self.metrics_[name].result()
                self.history_[name].append({
                    'chunk': int(chunk_index),
                    'chunk_accuracy': float(np.mean(pred == y_chunk)),
                    'cumulative_accuracy': float(metric_values['accuracy']),
                    'macro_f1': float(metric_values['f1_macro']),
                    'rolling_accuracy': float(metric_values['rolling_accuracy']),
                })
        return self.history_

    def summary(self) -> dict[str, dict]:
        return {
            name: {
                'chunks': len(rows),
                'final_accuracy': rows[-1]['cumulative_accuracy'] if rows else 0.0,
                'final_macro_f1': rows[-1]['macro_f1'] if rows else 0.0,
            }
            for name, rows in self.history_.items()
        }

    @staticmethod
    def _fit(model, X, y) -> None:
        if hasattr(model, 'partial_fit'):
            model.partial_fit(X, y)
        elif hasattr(model, 'fit'):
            model.fit(X, y)
        else:
            raise AttributeError('Models must provide fit() or partial_fit().')
