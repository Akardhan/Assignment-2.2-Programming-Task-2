"""Depth-limited decision tree classifier with streaming partial_fit support."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


Array = np.ndarray


@dataclass
class _Node:
    prediction: object
    proba: np.ndarray
    feature_index: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional['_Node'] = None
    right: Optional['_Node'] = None
    depth: int = 0

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


class DecisionTreeClassifier:
   

    def __init__(
        self,
        max_depth: int = 5,
        min_samples_split: int = 2,
        criterion: str = 'gini',
        max_features: int | float | str | None = None,
        max_thresholds: int = 32,
        max_samples: Optional[int] = 5000,
        random_state: Optional[int] = None,
    ) -> None:
        if max_depth < 0:
            raise ValueError('max_depth must be non-negative.')
        if min_samples_split < 2:
            raise ValueError('min_samples_split must be at least 2.')
        if criterion not in {'gini', 'entropy'}:
            raise ValueError("criterion must be 'gini' or 'entropy'.")
        if max_thresholds <= 0:
            raise ValueError('max_thresholds must be positive.')
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.max_features = max_features
        self.max_thresholds = max_thresholds
        self.max_samples = max_samples
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)
        self.root_: _Node | None = None
        self.classes_: np.ndarray | None = None
        self.n_features_in_: int | None = None
        self.feature_means_: np.ndarray | None = None
        self.feature_importances_: np.ndarray | None = None
        self.depth_: int = 0
        self.n_leaves_: int = 0
        self.node_count_: int = 0
        self._feature_importances_raw: np.ndarray | None = None
        self._buffer_X: Array | None = None
        self._buffer_y: Array | None = None

    def fit(self, X: Array, y: Array) -> 'DecisionTreeClassifier':
        X, y = self._validate_xy(X, y)
        self._buffer_X = X.copy()
        self._buffer_y = y.copy()
        self._trim_buffer()
        return self._rebuild()

    def partial_fit(self, X_chunk: Array, y_chunk: Array) -> 'DecisionTreeClassifier':
        X, y = self._validate_xy(X_chunk, y_chunk)
        if self._buffer_X is None:
            self._buffer_X = X.copy()
            self._buffer_y = y.copy()
        else:
            if X.shape[1] != self.n_features_in_:
                raise ValueError(f'Expected {self.n_features_in_} features, got {X.shape[1]}.')
            self._buffer_X = np.vstack([self._buffer_X, X])
            self._buffer_y = np.concatenate([self._buffer_y, y])
        self._trim_buffer()
        return self._rebuild()

    def predict(self, X: Array) -> Array:
        self._check_fitted()
        X = self._prepare_X(X)
        preds = [self._predict_one(row, self.root_) for row in X]
        return np.array(preds, dtype=self.classes_.dtype)

    def predict_proba(self, X: Array) -> Array:
        self._check_fitted()
        X = self._prepare_X(X)
        probs = [self._predict_proba_one(row, self.root_) for row in X]
        return np.vstack(probs)

    def score(self, X: Array, y: Array) -> float:
        y = np.asarray(y).ravel()
        pred = self.predict(X)
        if pred.shape[0] != y.shape[0]:
            raise ValueError('X and y length mismatch.')
        return float(np.mean(pred == y))

    def _validate_xy(self, X: Array, y: Array) -> tuple[Array, Array]:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.ndim != 2:
            raise ValueError('X must be a 1D or 2D numeric array.')
        y = np.asarray(y).ravel()
        if X.shape[0] != y.shape[0]:
            raise ValueError('X and y must contain the same number of samples.')
        if X.shape[0] == 0:
            raise ValueError('Cannot fit on an empty chunk.')
        if self.n_features_in_ is None:
            self.n_features_in_ = X.shape[1]
        elif X.shape[1] != self.n_features_in_:
            raise ValueError(f'Expected {self.n_features_in_} features, got {X.shape[1]}.')
        return X, y

    def _trim_buffer(self) -> None:
        if self.max_samples is not None and self._buffer_X.shape[0] > self.max_samples:
            self._buffer_X = self._buffer_X[-self.max_samples :]
            self._buffer_y = self._buffer_y[-self.max_samples :]

    def _rebuild(self) -> 'DecisionTreeClassifier':
        X = self._buffer_X
        y = self._buffer_y
        self.classes_ = np.unique(y)
        self.feature_means_ = np.nanmean(X, axis=0)
        self.feature_means_ = np.where(np.isnan(self.feature_means_), 0.0, self.feature_means_)
        X_clean = np.where(np.isnan(X), self.feature_means_, X)
        self._feature_importances_raw = np.zeros(X_clean.shape[1], dtype=float)
        self.root_ = self._build_tree(X_clean, y, depth=0)
        total_gain = float(np.sum(self._feature_importances_raw))
        if total_gain > 0:
            self.feature_importances_ = self._feature_importances_raw / total_gain
        else:
            self.feature_importances_ = np.zeros(X_clean.shape[1], dtype=float)
        self.depth_ = self._max_depth(self.root_)
        self.n_leaves_ = self._count_leaves(self.root_)
        self.node_count_ = self._count_nodes(self.root_)
        return self

    def _build_tree(self, X: Array, y: Array, depth: int) -> _Node:
        prediction = self._majority_class(y)
        proba = self._class_proba(y)
        node = _Node(prediction=prediction, proba=proba, depth=depth)
        if (
            depth >= self.max_depth
            or y.shape[0] < self.min_samples_split
            or np.unique(y).shape[0] == 1
        ):
            return node
        split = self._best_split(X, y)
        if split is None:
            return node
        feature_index, threshold, left_mask, gain = split
        self._feature_importances_raw[feature_index] += gain * y.shape[0]
        right_mask = ~left_mask
        if left_mask.sum() == 0 or right_mask.sum() == 0:
            return node
        node.feature_index = feature_index
        node.threshold = threshold
        node.left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        node.right = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        return node

    def _best_split(self, X: Array, y: Array):
        n_samples, n_features = X.shape
        parent_impurity = self._impurity(y)
        best_gain = 1e-12
        best = None
        feature_indices = self._sample_features(n_features)
        for j in feature_indices:
            values = X[:, j]
            thresholds = self._candidate_thresholds(values)
            for threshold in thresholds:
                left_mask = values <= threshold
                left_n = int(left_mask.sum())
                if left_n == 0 or left_n == n_samples:
                    continue
                right_n = n_samples - left_n
                gain = parent_impurity
                gain -= (left_n / n_samples) * self._impurity(y[left_mask])
                gain -= (right_n / n_samples) * self._impurity(y[~left_mask])
                if gain > best_gain:
                    best_gain = gain
                    best = (j, float(threshold), left_mask, float(gain))
        return best

    def _candidate_thresholds(self, values: Array) -> Array:
        unique = np.unique(values)
        if unique.shape[0] <= 1:
            return np.array([])
        if unique.shape[0] > self.max_thresholds:
            qs = np.linspace(0.0, 1.0, self.max_thresholds + 2)[1:-1]
            thresholds = np.quantile(unique, qs)
            return np.unique(thresholds)
        return (unique[:-1] + unique[1:]) / 2.0

    def _sample_features(self, n_features: int) -> Array:
        max_features = self.max_features
        if max_features is None:
            k = n_features
        elif isinstance(max_features, str):
            if max_features == 'sqrt':
                k = max(1, int(np.sqrt(n_features)))
            elif max_features == 'log2':
                k = max(1, int(np.log2(n_features)))
            else:
                raise ValueError("max_features string must be 'sqrt' or 'log2'.")
        elif isinstance(max_features, float):
            if not 0 < max_features <= 1:
                raise ValueError('float max_features must be in (0, 1].')
            k = max(1, int(np.ceil(max_features * n_features)))
        else:
            k = int(max_features)
        k = min(max(1, k), n_features)
        if k == n_features:
            return np.arange(n_features)
        return self.rng.choice(n_features, size=k, replace=False)

    def _impurity(self, y: Array) -> float:
        if y.shape[0] == 0:
            return 0.0
        counts = np.array([np.sum(y == c) for c in self.classes_], dtype=float)
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        if self.criterion == 'gini':
            return float(1.0 - np.sum(probs ** 2))
        return float(-np.sum(probs * np.log2(probs)))

    def _majority_class(self, y: Array):
        counts = np.array([np.sum(y == c) for c in self.classes_])
        max_count = counts.max()
        # Deterministic tie resolution by order in self.classes_.
        return self.classes_[np.flatnonzero(counts == max_count)[0]]

    def _class_proba(self, y: Array) -> Array:
        counts = np.array([np.sum(y == c) for c in self.classes_], dtype=float)
        total = counts.sum()
        if total == 0:
            return np.ones(len(self.classes_)) / len(self.classes_)
        return counts / total


    def _max_depth(self, node: _Node | None) -> int:
        if node is None or node.is_leaf:
            return 0
        return 1 + max(self._max_depth(node.left), self._max_depth(node.right))

    def _count_leaves(self, node: _Node | None) -> int:
        if node is None:
            return 0
        if node.is_leaf:
            return 1
        return self._count_leaves(node.left) + self._count_leaves(node.right)

    def _count_nodes(self, node: _Node | None) -> int:
        if node is None:
            return 0
        return 1 + self._count_nodes(node.left) + self._count_nodes(node.right)

    def describe(self) -> dict:
        """Return marker-friendly model summary values."""
        self._check_fitted()
        return {
            'n_features': int(self.n_features_in_),
            'n_classes': int(len(self.classes_)),
            'depth': int(self.depth_),
            'n_leaves': int(self.n_leaves_),
            'node_count': int(self.node_count_),
            'criterion': self.criterion,
            'max_depth': int(self.max_depth),
        }

    def _prepare_X(self, X: Array) -> Array:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.ndim != 2:
            raise ValueError('X must be a 1D or 2D numeric array.')
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f'Expected {self.n_features_in_} features, got {X.shape[1]}.')
        return np.where(np.isnan(X), self.feature_means_, X)

    def _predict_one(self, row: Array, node: _Node):
        while not node.is_leaf:
            if row[node.feature_index] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.prediction

    def _predict_proba_one(self, row: Array, node: _Node) -> Array:
        while not node.is_leaf:
            if row[node.feature_index] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.proba.copy()

    def _check_fitted(self) -> None:
        if self.root_ is None:
            raise RuntimeError('DecisionTreeClassifier is not fitted yet.')
