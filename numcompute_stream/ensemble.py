"""Tree ensemble classifier with streaming adaptation."""

from __future__ import annotations

from typing import Optional

import numpy as np

from .tree import DecisionTreeClassifier


Array = np.ndarray


class EnsembleClassifier:
    """Streaming tree ensemble using bagging or random forest style sampling.

    Parameters are intentionally similar to ``DecisionTreeClassifier``. On each
    ``partial_fit`` call, the ensemble appends the new chunk to a bounded replay
    buffer and refits each tree on a bootstrap sample. For ``method='random_forest'``
    each tree also uses random feature subsampling at splits.
    """

    def __init__(
        self,
        n_estimators: int = 7,
        method: str = 'random_forest',
        max_depth: int = 5,
        min_samples_split: int = 2,
        criterion: str = 'gini',
        max_features: int | float | str | None = 'sqrt',
        bootstrap: bool = True,
        sample_ratio: float = 1.0,
        max_samples: Optional[int] = 5000,
        random_state: Optional[int] = None,
        oob_score: bool = False,
    ) -> None:
        if n_estimators <= 0:
            raise ValueError('n_estimators must be positive.')
        if method not in {'bagging', 'random_forest'}:
            raise ValueError("method must be 'bagging' or 'random_forest'.")
        if max_depth <= 0:
            raise ValueError('max_depth must be positive.')
        if min_samples_split < 2:
            raise ValueError('min_samples_split must be at least 2.')
        if criterion not in {'gini', 'entropy'}:
            raise ValueError("criterion must be 'gini' or 'entropy'.")
        if not 0 < sample_ratio <= 1:
            raise ValueError('sample_ratio must be in (0, 1].')
        if max_samples is not None and max_samples <= 0:
            raise ValueError('max_samples must be positive when provided.')
        self.n_estimators = n_estimators
        self.method = method
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.sample_ratio = sample_ratio
        self.max_samples = max_samples
        self.random_state = random_state
        self.oob_score = oob_score
        self.rng = np.random.default_rng(random_state)
        self.estimators_: list[DecisionTreeClassifier] = []
        self.classes_: np.ndarray | None = None
        self.n_features_in_: int | None = None
        self.feature_importances_: np.ndarray | None = None
        self.oob_score_: float | None = None
        self._buffer_X: Array | None = None
        self._buffer_y: Array | None = None

    def fit(self, X: Array, y: Array) -> 'EnsembleClassifier':
        self._reset_state()
        X, y = self._validate_xy(X, y)
        self._buffer_X = X.copy()
        self._buffer_y = y.copy()
        self._trim_buffer()
        return self._rebuild()

    def partial_fit(self, X_chunk: Array, y_chunk: Array) -> 'EnsembleClassifier':
        X, y = self._validate_xy(X_chunk, y_chunk)
        if self._buffer_X is None:
            self._buffer_X = X.copy()
            self._buffer_y = y.copy()
        else:
            self._buffer_X = np.vstack([self._buffer_X, X])
            self._buffer_y = np.concatenate([self._buffer_y, y])
        self._trim_buffer()
        return self._rebuild()

    def predict(self, X: Array) -> Array:
        self._check_fitted()
        X = self._prepare_X(X)
        preds = np.vstack([est.predict(X) for est in self.estimators_]).T
        counts = np.vstack([np.sum(preds == cls, axis=1) for cls in self.classes_]).T
        return self.classes_[np.argmax(counts, axis=1)]

    def predict_proba(self, X: Array) -> Array:
        self._check_fitted()
        X = self._prepare_X(X)
        probs = np.zeros((X.shape[0], len(self.classes_)), dtype=float)
        for est in self.estimators_:
            est_probs = est.predict_proba(X)
            est_classes = est.classes_
            for j, cls in enumerate(est_classes):
                target = np.where(self.classes_ == cls)[0][0]
                probs[:, target] += est_probs[:, j]
        return probs / len(self.estimators_)

    def score(self, X: Array, y: Array) -> float:
        y = np.asarray(y).ravel()
        pred = self.predict(X)
        if y.shape[0] != pred.shape[0]:
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

    def _rebuild(self) -> 'EnsembleClassifier':
        X = self._buffer_X
        y = self._buffer_y
        self.classes_ = np.unique(y)
        self.estimators_ = []
        n = X.shape[0]
        sample_n = max(1, int(round(n * self.sample_ratio)))
        tree_max_features = self.max_features if self.method == 'random_forest' else None
        oob_votes = np.zeros((n, len(self.classes_)), dtype=float)
        oob_counts = np.zeros(n, dtype=int)
        class_to_index = {label: pos for pos, label in enumerate(self.classes_)}
        for i in range(self.n_estimators):
            if self.bootstrap:
                idx = self.rng.choice(n, size=sample_n, replace=True)
            else:
                idx = self.rng.choice(n, size=sample_n, replace=False)
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                criterion=self.criterion,
                max_features=tree_max_features,
                max_samples=None,
                random_state=None if self.random_state is None else self.random_state + i + len(self.estimators_),
            )
            tree.fit(X[idx], y[idx])
            self.estimators_.append(tree)

            if self.oob_score and self.bootstrap:
                in_bag = np.zeros(n, dtype=bool)
                in_bag[idx] = True
                oob_idx = np.flatnonzero(~in_bag)
                if oob_idx.size:
                    pred = tree.predict(X[oob_idx])
                    for row_pos, label in zip(oob_idx, pred):
                        oob_votes[row_pos, class_to_index[label]] += 1.0
                        oob_counts[row_pos] += 1

        self.feature_importances_ = np.mean(
            np.vstack([tree.feature_importances_ for tree in self.estimators_]),
            axis=0,
        )
        total_importance = float(np.sum(self.feature_importances_))
        if total_importance > 0:
            self.feature_importances_ = self.feature_importances_ / total_importance

        if self.oob_score and self.bootstrap:
            valid = oob_counts > 0
            if np.any(valid):
                oob_pred = self.classes_[np.argmax(oob_votes[valid], axis=1)]
                self.oob_score_ = float(np.mean(oob_pred == y[valid]))
            else:
                self.oob_score_ = None
        else:
            self.oob_score_ = None
        return self


    def describe(self) -> dict:
        """Return marker-friendly ensemble summary values."""
        self._check_fitted()
        depths = np.array([tree.depth_ for tree in self.estimators_], dtype=float)
        leaves = np.array([tree.n_leaves_ for tree in self.estimators_], dtype=float)
        return {
            'n_estimators': int(self.n_estimators),
            'method': self.method,
            'n_features': int(self.n_features_in_),
            'n_classes': int(len(self.classes_)),
            'mean_tree_depth': float(np.mean(depths)),
            'mean_tree_leaves': float(np.mean(leaves)),
            'oob_score': self.oob_score_,
        }

    def summary(self) -> dict:
        """Alias for ``describe`` used by demos and reports."""
        return self.describe()

    def _prepare_X(self, X: Array) -> Array:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.ndim != 2:
            raise ValueError('X must be a 1D or 2D numeric array.')
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f'Expected {self.n_features_in_} features, got {X.shape[1]}.')
        return X

    def _reset_state(self) -> None:
        self.rng = np.random.default_rng(self.random_state)
        self.estimators_ = []
        self.classes_ = None
        self.n_features_in_ = None
        self.feature_importances_ = None
        self.oob_score_ = None
        self._buffer_X = None
        self._buffer_y = None

    def _check_fitted(self) -> None:
        if not self.estimators_:
            raise RuntimeError('EnsembleClassifier is not fitted yet.')


class BoostingClassifier:
    """Streaming AdaBoost-style ensemble built from NumPy decision trees.

    This class adds a second ensemble family beyond bagging/random forests.
    Because the base ``DecisionTreeClassifier`` is intentionally NumPy-only and
    does not use sample weights directly, weighted learning is approximated by
    drawing each weak learner's training set from the replay buffer according to
    the current boosting weights. The estimator is rebuilt on every
    ``partial_fit`` call from a bounded buffer, matching the streaming API used
    elsewhere in the package.

    Parameters
    ----------
    n_estimators:
        Maximum number of weak decision trees.
    max_depth:
        Depth of each weak tree. A value of 1 gives decision stumps.
    learning_rate:
        Shrinks each weak learner weight.
    max_samples:
        Maximum rows retained in the streaming replay buffer.
    random_state:
        Seed for reproducible weighted resampling.
    """

    def __init__(
        self,
        n_estimators: int = 25,
        max_depth: int = 1,
        learning_rate: float = 1.0,
        min_samples_split: int = 2,
        criterion: str = 'gini',
        max_samples: Optional[int] = 5000,
        random_state: Optional[int] = None,
    ) -> None:
        if n_estimators <= 0:
            raise ValueError('n_estimators must be positive.')
        if max_depth <= 0:
            raise ValueError('max_depth must be positive.')
        if learning_rate <= 0:
            raise ValueError('learning_rate must be positive.')
        if min_samples_split < 2:
            raise ValueError('min_samples_split must be at least 2.')
        if criterion not in {'gini', 'entropy'}:
            raise ValueError("criterion must be 'gini' or 'entropy'.")
        if max_samples is not None and max_samples <= 0:
            raise ValueError('max_samples must be positive when provided.')
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.max_samples = max_samples
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)
        self.estimators_: list[DecisionTreeClassifier] = []
        self.estimator_weights_: np.ndarray | None = None
        self.estimator_errors_: np.ndarray | None = None
        self.classes_: np.ndarray | None = None
        self.n_features_in_: int | None = None
        self.feature_importances_: np.ndarray | None = None
        self._buffer_X: Array | None = None
        self._buffer_y: Array | None = None

    def fit(self, X: Array, y: Array) -> 'BoostingClassifier':
        self._reset_state()
        X, y = self._validate_xy(X, y)
        self._buffer_X = X.copy()
        self._buffer_y = y.copy()
        self._trim_buffer()
        return self._rebuild()

    def partial_fit(self, X_chunk: Array, y_chunk: Array) -> 'BoostingClassifier':
        X, y = self._validate_xy(X_chunk, y_chunk)
        if self._buffer_X is None:
            self._buffer_X = X.copy()
            self._buffer_y = y.copy()
        else:
            self._buffer_X = np.vstack([self._buffer_X, X])
            self._buffer_y = np.concatenate([self._buffer_y, y])
        self._trim_buffer()
        return self._rebuild()

    def predict(self, X: Array) -> Array:
        self._check_fitted()
        X = self._prepare_X(X)
        scores = self._decision_scores(X)
        return self.classes_[np.argmax(scores, axis=1)]

    def predict_proba(self, X: Array) -> Array:
        self._check_fitted()
        X = self._prepare_X(X)
        # A weighted average of tree probabilities is more stable than a hard
        # vote-only softmax for tiny chunks and ties.
        weights = self.estimator_weights_.astype(float)
        total_weight = float(np.sum(weights))
        probs = np.zeros((X.shape[0], len(self.classes_)), dtype=float)
        for tree, alpha in zip(self.estimators_, weights):
            tree_probs = tree.predict_proba(X)
            for j, cls in enumerate(tree.classes_):
                target = np.where(self.classes_ == cls)[0][0]
                probs[:, target] += alpha * tree_probs[:, j]
        if total_weight <= 0:
            return np.ones_like(probs) / probs.shape[1]
        probs /= total_weight
        row_sums = probs.sum(axis=1, keepdims=True)
        return np.divide(probs, row_sums, out=np.ones_like(probs) / probs.shape[1], where=row_sums > 0)

    def score(self, X: Array, y: Array) -> float:
        y = np.asarray(y).ravel()
        pred = self.predict(X)
        if y.shape[0] != pred.shape[0]:
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
            self._buffer_X = self._buffer_X[-self.max_samples:]
            self._buffer_y = self._buffer_y[-self.max_samples:]

    def _rebuild(self) -> 'BoostingClassifier':
        X = self._buffer_X
        y = self._buffer_y
        self.classes_ = np.unique(y)
        n = X.shape[0]
        n_classes = len(self.classes_)
        weights = np.ones(n, dtype=float) / n
        estimators: list[DecisionTreeClassifier] = []
        alphas: list[float] = []
        errors: list[float] = []

        if n_classes == 1:
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                criterion=self.criterion,
                max_samples=None,
                random_state=self.random_state,
            ).fit(X, y)
            self.estimators_ = [tree]
            self.estimator_weights_ = np.array([1.0], dtype=float)
            self.estimator_errors_ = np.array([0.0], dtype=float)
            self.feature_importances_ = tree.feature_importances_.copy()
            return self

        for i in range(self.n_estimators):
            idx = self.rng.choice(n, size=n, replace=True, p=weights)
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                criterion=self.criterion,
                max_samples=None,
                random_state=None if self.random_state is None else self.random_state + i,
            ).fit(X[idx], y[idx])
            pred = tree.predict(X)
            incorrect = pred != y
            error = float(np.dot(weights, incorrect.astype(float)))
            error = min(max(error, 1e-12), 1.0 - 1e-12)

            # SAMME requires weak learners better than random guessing.
            random_error = 1.0 - (1.0 / n_classes)
            if error >= random_error:
                if not estimators:
                    estimators.append(tree)
                    alphas.append(1e-6)
                    errors.append(error)
                break

            alpha = self.learning_rate * (np.log((1.0 - error) / error) + np.log(n_classes - 1))
            alpha = float(np.clip(alpha, 1e-6, 10.0))
            estimators.append(tree)
            alphas.append(alpha)
            errors.append(error)

            weights *= np.exp(alpha * incorrect.astype(float))
            weight_sum = weights.sum()
            if weight_sum <= 0 or not np.isfinite(weight_sum):
                weights = np.ones(n, dtype=float) / n
            else:
                weights /= weight_sum

            if error <= 1e-12:
                break

        self.estimators_ = estimators
        self.estimator_weights_ = np.array(alphas, dtype=float)
        self.estimator_errors_ = np.array(errors, dtype=float)
        if self.estimators_:
            weighted_importances = np.zeros(self.n_features_in_, dtype=float)
            total_alpha = float(np.sum(self.estimator_weights_))
            for tree, alpha in zip(self.estimators_, self.estimator_weights_):
                weighted_importances += alpha * tree.feature_importances_
            if total_alpha > 0:
                weighted_importances /= total_alpha
            s = float(weighted_importances.sum())
            self.feature_importances_ = weighted_importances / s if s > 0 else weighted_importances
        else:
            self.feature_importances_ = np.zeros(self.n_features_in_, dtype=float)
        return self

    def _decision_scores(self, X: Array) -> Array:
        scores = np.zeros((X.shape[0], len(self.classes_)), dtype=float)
        for tree, alpha in zip(self.estimators_, self.estimator_weights_):
            pred = tree.predict(X)
            for j, cls in enumerate(self.classes_):
                scores[:, j] += alpha * (pred == cls)
        return scores

    def _prepare_X(self, X: Array) -> Array:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.ndim != 2:
            raise ValueError('X must be a 1D or 2D numeric array.')
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f'Expected {self.n_features_in_} features, got {X.shape[1]}.')
        return X

    def describe(self) -> dict:
        """Return marker-friendly boosting summary values."""
        self._check_fitted()
        depths = np.array([tree.depth_ for tree in self.estimators_], dtype=float)
        leaves = np.array([tree.n_leaves_ for tree in self.estimators_], dtype=float)
        return {
            'n_estimators_requested': int(self.n_estimators),
            'n_estimators_used': int(len(self.estimators_)),
            'method': 'boosting_samme',
            'n_features': int(self.n_features_in_),
            'n_classes': int(len(self.classes_)),
            'mean_tree_depth': float(np.mean(depths)) if depths.size else 0.0,
            'mean_tree_leaves': float(np.mean(leaves)) if leaves.size else 0.0,
            'mean_estimator_error': float(np.mean(self.estimator_errors_)) if self.estimator_errors_.size else 0.0,
        }

    def summary(self) -> dict:
        """Alias for ``describe`` used by demos and reports."""
        return self.describe()

    def _reset_state(self) -> None:
        self.rng = np.random.default_rng(self.random_state)
        self.estimators_ = []
        self.estimator_weights_ = None
        self.estimator_errors_ = None
        self.classes_ = None
        self.n_features_in_ = None
        self.feature_importances_ = None
        self._buffer_X = None
        self._buffer_y = None

    def _check_fitted(self) -> None:
        if not self.estimators_:
            raise RuntimeError('BoostingClassifier is not fitted yet.')
