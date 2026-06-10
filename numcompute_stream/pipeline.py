"""Simple streaming pipeline for transformers and final estimators."""

from __future__ import annotations

from typing import Iterable, List, Tuple

import numpy as np


class Pipeline:
    """Chain transformers and a final model with consistent fit/predict APIs.

    Each intermediate step must provide ``transform`` and optionally
    ``partial_fit``. The final step must provide ``fit``/``partial_fit`` and
    ``predict``. During ``partial_fit``, transformers are updated first and the
    transformed chunk is then passed to the model.
    """

    def __init__(self, steps: Iterable[Tuple[str, object]]) -> None:
        self.steps: List[Tuple[str, object]] = list(steps)
        if not self.steps:
            raise ValueError('Pipeline requires at least one step.')
        names = [name for name, _ in self.steps]
        if len(names) != len(set(names)):
            raise ValueError('Pipeline step names must be unique.')

    @property
    def named_steps(self) -> dict:
        return dict(self.steps)

    def fit(self, X, y=None) -> 'Pipeline':
        Xt = X
        for name, step in self.steps[:-1]:
            if hasattr(step, 'fit_transform'):
                Xt = step.fit_transform(Xt, y)
            else:
                step.fit(Xt, y)
                Xt = step.transform(Xt)
        final = self.steps[-1][1]
        if y is None:
            final.fit(Xt)
        else:
            final.fit(Xt, y)
        return self

    def partial_fit(self, X, y=None) -> 'Pipeline':
        Xt = X
        for name, step in self.steps[:-1]:
            if hasattr(step, 'partial_fit'):
                step.partial_fit(Xt, y)
            elif hasattr(step, 'fit'):
                step.fit(Xt, y)
            if not hasattr(step, 'transform'):
                raise AttributeError(f'Step {name!r} does not provide transform().')
            Xt = step.transform(Xt)
        final = self.steps[-1][1]
        if hasattr(final, 'partial_fit'):
            if y is None:
                final.partial_fit(Xt)
            else:
                final.partial_fit(Xt, y)
        elif hasattr(final, 'fit'):
            if y is None:
                final.fit(Xt)
            else:
                final.fit(Xt, y)
        else:
            raise AttributeError('Final pipeline step must provide fit() or partial_fit().')
        return self

    def transform(self, X):
        Xt = X
        for name, step in self.steps:
            if not hasattr(step, 'transform'):
                raise AttributeError(f'Step {name!r} does not provide transform().')
            Xt = step.transform(Xt)
        return Xt

    def predict(self, X):
        Xt = X
        for name, step in self.steps[:-1]:
            if not hasattr(step, 'transform'):
                raise AttributeError(f'Step {name!r} does not provide transform().')
            Xt = step.transform(Xt)
        final = self.steps[-1][1]
        if not hasattr(final, 'predict'):
            raise AttributeError('Final pipeline step must provide predict().')
        return final.predict(Xt)

    def predict_proba(self, X):
        Xt = X
        for name, step in self.steps[:-1]:
            Xt = step.transform(Xt)
        final = self.steps[-1][1]
        if not hasattr(final, 'predict_proba'):
            raise AttributeError('Final pipeline step does not provide predict_proba().')
        return final.predict_proba(Xt)

    def score(self, X, y) -> float:
        pred = self.predict(X)
        y = np.asarray(y).ravel()
        if pred.shape[0] != y.shape[0]:
            raise ValueError('X and y length mismatch.')
        return float(np.mean(pred == y))
