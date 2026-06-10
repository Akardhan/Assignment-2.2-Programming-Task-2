"""Streaming classification metrics."""

from __future__ import annotations

from collections import deque
from typing import Dict, Iterable, Optional

import numpy as np


Array = np.ndarray


def _as_1d(a: Array) -> Array:
    a = np.asarray(a)
    if a.ndim != 1:
        a = a.ravel()
    return a


class StreamingAccuracy:
    """Accumulate accuracy over chunks."""

    def __init__(self) -> None:
        self.correct_ = 0
        self.total_ = 0

    def update(self, y_true: Array, y_pred: Array) -> 'StreamingAccuracy':
        y_true = _as_1d(y_true)
        y_pred = _as_1d(y_pred)
        if y_true.shape[0] != y_pred.shape[0]:
            raise ValueError('y_true and y_pred must have the same length.')
        self.correct_ += int(np.sum(y_true == y_pred))
        self.total_ += int(y_true.shape[0])
        return self

    def result(self) -> float:
        return float(self.correct_ / self.total_) if self.total_ else 0.0

    def reset(self) -> None:
        self.correct_ = 0
        self.total_ = 0


class StreamingConfusionMatrix:
    """Accumulate a confusion matrix over time."""

    def __init__(self, classes: Optional[Iterable] = None) -> None:
        self.classes_: np.ndarray | None = np.array(list(classes)) if classes is not None else None
        self.matrix_: np.ndarray | None = None

    def update(self, y_true: Array, y_pred: Array) -> 'StreamingConfusionMatrix':
        y_true = _as_1d(y_true)
        y_pred = _as_1d(y_pred)
        if y_true.shape[0] != y_pred.shape[0]:
            raise ValueError('y_true and y_pred must have the same length.')
        seen = np.unique(np.concatenate([y_true, y_pred]))
        if self.classes_ is None:
            self.classes_ = seen
            self.matrix_ = np.zeros((len(self.classes_), len(self.classes_)), dtype=int)
        else:
            if self.matrix_ is None:
                self.matrix_ = np.zeros((len(self.classes_), len(self.classes_)), dtype=int)
            missing = [c for c in seen if c not in set(self.classes_)]
            if missing:
                old_classes = self.classes_.copy()
                old_matrix = self.matrix_.copy()
                self.classes_ = np.concatenate([self.classes_, np.array(missing, dtype=old_classes.dtype)])
                self.matrix_ = np.zeros((len(self.classes_), len(self.classes_)), dtype=int)
                self.matrix_[: len(old_classes), : len(old_classes)] = old_matrix
        lookup = {label: i for i, label in enumerate(self.classes_)}
        true_idx = np.array([lookup[label] for label in y_true], dtype=int)
        pred_idx = np.array([lookup[label] for label in y_pred], dtype=int)
        np.add.at(self.matrix_, (true_idx, pred_idx), 1)
        return self

    def result(self) -> Array:
        if self.matrix_ is None:
            return np.zeros((0, 0), dtype=int)
        return self.matrix_.copy()

    def reset(self) -> None:
        if self.classes_ is None:
            self.matrix_ = None
        else:
            self.matrix_ = np.zeros((len(self.classes_), len(self.classes_)), dtype=int)


class _PRFBase:
    def __init__(self, classes: Optional[Iterable] = None, average: str = 'macro') -> None:
        if average not in {'macro', 'micro'}:
            raise ValueError("average must be 'macro' or 'micro'.")
        self.average = average
        self.cm = StreamingConfusionMatrix(classes)

    def update(self, y_true: Array, y_pred: Array):
        self.cm.update(y_true, y_pred)
        return self

    def reset(self) -> None:
        self.cm.reset()

    def _counts(self):
        matrix = self.cm.result().astype(float)
        tp = np.diag(matrix)
        fp = matrix.sum(axis=0) - tp
        fn = matrix.sum(axis=1) - tp
        return tp, fp, fn


class StreamingPrecision(_PRFBase):
    """Streaming precision with macro or micro averaging."""

    def result(self) -> float:
        tp, fp, _ = self._counts()
        if tp.size == 0:
            return 0.0
        if self.average == 'micro':
            denom = tp.sum() + fp.sum()
            return float(tp.sum() / denom) if denom else 0.0
        values = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
        return float(np.mean(values))


class StreamingRecall(_PRFBase):
    """Streaming recall with macro or micro averaging."""

    def result(self) -> float:
        tp, _, fn = self._counts()
        if tp.size == 0:
            return 0.0
        if self.average == 'micro':
            denom = tp.sum() + fn.sum()
            return float(tp.sum() / denom) if denom else 0.0
        values = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
        return float(np.mean(values))


class StreamingF1(_PRFBase):
    """Streaming F1 score with macro or micro averaging."""

    def result(self) -> float:
        precision = StreamingPrecision(average=self.average)
        precision.cm.classes_ = self.cm.classes_
        precision.cm.matrix_ = self.cm.result()
        recall = StreamingRecall(average=self.average)
        recall.cm.classes_ = self.cm.classes_
        recall.cm.matrix_ = self.cm.result()
        p = precision.result()
        r = recall.result()
        return float(2 * p * r / (p + r)) if (p + r) else 0.0


class RollingAccuracy:
    """Accuracy over the most recent ``window_size`` predictions."""

    def __init__(self, window_size: int = 100) -> None:
        if window_size <= 0:
            raise ValueError('window_size must be positive.')
        self.window_size = window_size
        self.items = deque(maxlen=window_size)

    def update(self, y_true: Array, y_pred: Array) -> 'RollingAccuracy':
        y_true = _as_1d(y_true)
        y_pred = _as_1d(y_pred)
        if y_true.shape[0] != y_pred.shape[0]:
            raise ValueError('y_true and y_pred must have the same length.')
        self.items.extend((y_true == y_pred).astype(int).tolist())
        return self

    def result(self) -> float:
        if not self.items:
            return 0.0
        return float(np.mean(np.array(self.items)))

    def reset(self) -> None:
        self.items.clear()




class RollingConfusionMatrix:
    """Confusion matrix over the most recent ``window_size`` predictions."""

    def __init__(self, classes: Optional[Iterable] = None, window_size: int = 100) -> None:
        if window_size <= 0:
            raise ValueError('window_size must be positive.')
        self.window_size = window_size
        self.classes_: np.ndarray | None = np.array(list(classes)) if classes is not None else None
        self.items = deque(maxlen=window_size)

    def update(self, y_true: Array, y_pred: Array) -> 'RollingConfusionMatrix':
        y_true = _as_1d(y_true)
        y_pred = _as_1d(y_pred)
        if y_true.shape[0] != y_pred.shape[0]:
            raise ValueError('y_true and y_pred must have the same length.')
        seen = np.unique(np.concatenate([y_true, y_pred]))
        if self.classes_ is None:
            self.classes_ = seen
        else:
            missing = [c for c in seen if c not in set(self.classes_)]
            if missing:
                self.classes_ = np.concatenate([self.classes_, np.array(missing, dtype=self.classes_.dtype)])
        self.items.extend(list(zip(y_true.tolist(), y_pred.tolist())))
        return self

    def result(self) -> Array:
        if self.classes_ is None:
            return np.zeros((0, 0), dtype=int)
        matrix = np.zeros((len(self.classes_), len(self.classes_)), dtype=int)
        lookup = {label: i for i, label in enumerate(self.classes_)}
        if self.items:
            true_idx = np.array([lookup[t] for t, _ in self.items], dtype=int)
            pred_idx = np.array([lookup[p] for _, p in self.items], dtype=int)
            np.add.at(matrix, (true_idx, pred_idx), 1)
        return matrix

    def reset(self) -> None:
        self.items.clear()


class StreamingAUC:
    """Binary AUC accumulator.

    Stores a bounded history of labels and scores, then computes the rank-based
    ROC AUC. For memory-sensitive scenarios set ``max_points``.
    """

    def __init__(self, positive_label=1, max_points: Optional[int] = None) -> None:
        self.positive_label = positive_label
        self.max_points = max_points
        self.y_true_: list = []
        self.y_score_: list = []

    def update(self, y_true: Array, y_score: Array) -> 'StreamingAUC':
        y_true = _as_1d(y_true)
        y_score = _as_1d(y_score).astype(float)
        if y_true.shape[0] != y_score.shape[0]:
            raise ValueError('y_true and y_score must have the same length.')
        self.y_true_.extend((y_true == self.positive_label).astype(int).tolist())
        self.y_score_.extend(y_score.tolist())
        if self.max_points is not None and len(self.y_true_) > self.max_points:
            self.y_true_ = self.y_true_[-self.max_points :]
            self.y_score_ = self.y_score_[-self.max_points :]
        return self

    def result(self) -> float:
        y = np.array(self.y_true_, dtype=int)
        scores = np.array(self.y_score_, dtype=float)
        pos = np.sum(y == 1)
        neg = np.sum(y == 0)
        if pos == 0 or neg == 0:
            return 0.0
        order = np.argsort(scores)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(1, len(scores) + 1)
        # Average ranks for ties.
        unique_scores, inverse = np.unique(scores, return_inverse=True)
        for group in range(len(unique_scores)):
            mask = inverse == group
            if np.sum(mask) > 1:
                ranks[mask] = np.mean(ranks[mask])
        sum_pos_ranks = np.sum(ranks[y == 1])
        auc = (sum_pos_ranks - pos * (pos + 1) / 2) / (pos * neg)
        return float(auc)

    def reset(self) -> None:
        self.y_true_.clear()
        self.y_score_.clear()


class ClassificationMetrics:
    """Convenience wrapper that updates multiple streaming metrics together."""

    def __init__(self, classes: Optional[Iterable] = None, rolling_window: int = 100) -> None:
        self.metrics = {
            'accuracy': StreamingAccuracy(),
            'precision_macro': StreamingPrecision(classes=classes, average='macro'),
            'recall_macro': StreamingRecall(classes=classes, average='macro'),
            'f1_macro': StreamingF1(classes=classes, average='macro'),
            'rolling_accuracy': RollingAccuracy(window_size=rolling_window),
        }
        self.confusion_matrix = StreamingConfusionMatrix(classes=classes)
        self.rolling_confusion_matrix = RollingConfusionMatrix(classes=classes, window_size=rolling_window)

    def update(self, y_true: Array, y_pred: Array) -> 'ClassificationMetrics':
        for metric in self.metrics.values():
            metric.update(y_true, y_pred)
        self.confusion_matrix.update(y_true, y_pred)
        self.rolling_confusion_matrix.update(y_true, y_pred)
        return self

    def result(self) -> Dict[str, float | Array]:
        out = {name: metric.result() for name, metric in self.metrics.items()}
        out['confusion_matrix'] = self.confusion_matrix.result()
        out['rolling_confusion_matrix'] = self.rolling_confusion_matrix.result()
        return out

    def reset(self) -> None:
        for metric in self.metrics.values():
            metric.reset()
        self.confusion_matrix.reset()
        self.rolling_confusion_matrix.reset()
