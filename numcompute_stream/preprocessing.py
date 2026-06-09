""" Streaming preprocessing transformers implemented with NumPy only. """

from __future__ import annotations

from typing import List

import numpy as np


Array = np.ndarray


def _ensure_2d(X: Array) -> Array:
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.ndim != 2:
        raise ValueError("X must be a 1D or 2D numeric array.")
    return X


class StandardScaler:
    """ Incremental standardisation using running mean and variance.

    NaN values are ignored while updating statistics. During transform, NaNs are
    replaced by the learned mean before scaling. Zero-variance features are left
    unchanged by using scale value 1.
    
    """

    def __init__(self) -> None:
        self.n_seen_: np.ndarray | None = None
        self.mean_: np.ndarray | None = None
        self.M2_: np.ndarray | None = None
        self.var_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.n_features_in_: int | None = None

    def partial_fit(self, X: Array, y: Array | None = None) -> "StandardScaler":
        X = _ensure_2d(X)

        if self.n_features_in_ is None:
            self.n_features_in_ = X.shape[1]
            self.n_seen_ = np.zeros(X.shape[1], dtype=float)
            self.mean_ = np.zeros(X.shape[1], dtype=float)
            self.M2_ = np.zeros(X.shape[1], dtype=float)

        elif X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {X.shape[1]}."
            )

        mask = ~np.isnan(X)
        counts = mask.sum(axis=0).astype(float)

        safe_X = np.where(mask, X, 0.0)
        sums = safe_X.sum(axis=0)

        chunk_mean = np.divide(
            sums,
            counts,
            out=np.zeros_like(sums),
            where=counts > 0,
        )

        centered = np.where(mask, X - chunk_mean, 0.0)
        chunk_M2 = np.sum(centered * centered, axis=0)

        total = self.n_seen_ + counts
        delta = chunk_mean - self.mean_
        valid = counts > 0

        self.mean_[valid] = (
            self.mean_[valid]
            + delta[valid] * counts[valid] / total[valid]
        )

        self.M2_[valid] = (
            self.M2_[valid]
            + chunk_M2[valid]
            + (delta[valid] ** 2)
            * self.n_seen_[valid]
            * counts[valid]
            / total[valid]
        )

        self.n_seen_[valid] = total[valid]

        self.var_ = np.divide(
            self.M2_,
            self.n_seen_,
            out=np.zeros_like(self.M2_),
            where=self.n_seen_ > 0,
        )

        self.scale_ = np.sqrt(self.var_)
        self.scale_ = np.where(self.scale_ > 1e-12, self.scale_, 1.0)

        return self

    def fit(self, X: Array, y: Array | None = None) -> "StandardScaler":
        self.__init__()
        return self.partial_fit(X, y)

    def transform(self, X: Array) -> Array:
        self._check_fitted()
        X = _ensure_2d(X)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {X.shape[1]}."
            )

        filled = np.where(np.isnan(X), self.mean_, X)
        return (filled - self.mean_) / self.scale_

    def fit_transform(self, X: Array, y: Array | None = None) -> Array:
        return self.fit(X, y).transform(X)

    def _check_fitted(self) -> None:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("StandardScaler is not fitted yet.")


class SimpleImputer:
    """Streaming numeric imputer.

    Supports strategy='mean' and strategy='constant'. For the mean strategy,
    missing-value estimates are updated chunk by chunk.
    """

    def __init__(self, strategy: str = "mean", fill_value: float = 0.0) -> None:
        if strategy not in {"mean", "constant"}:
            raise ValueError("strategy must be 'mean' or 'constant'.")

        self.strategy = strategy
        self.fill_value = fill_value
        self.n_features_in_: int | None = None
        self.counts_: np.ndarray | None = None
        self.sums_: np.ndarray | None = None
        self.statistics_: np.ndarray | None = None

    def partial_fit(self, X: Array, y: Array | None = None) -> "SimpleImputer":
        X = _ensure_2d(X)

        if self.n_features_in_ is None:
            self.n_features_in_ = X.shape[1]
            self.counts_ = np.zeros(X.shape[1], dtype=float)
            self.sums_ = np.zeros(X.shape[1], dtype=float)

        elif X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {X.shape[1]}."
            )

        if self.strategy == "mean":
            mask = ~np.isnan(X)

            self.counts_ += mask.sum(axis=0)
            self.sums_ += np.where(mask, X, 0.0).sum(axis=0)

            self.statistics_ = np.divide(
                self.sums_,
                self.counts_,
                out=np.full_like(self.sums_, self.fill_value, dtype=float),
                where=self.counts_ > 0,
            )

        else:
            self.statistics_ = np.full(X.shape[1], self.fill_value, dtype=float)

        return self

    def fit(self, X: Array, y: Array | None = None) -> "SimpleImputer":
        self.__init__(self.strategy, self.fill_value)
        return self.partial_fit(X, y)

    def transform(self, X: Array) -> Array:
        self._check_fitted()
        X = _ensure_2d(X)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {X.shape[1]}."
            )

        return np.where(np.isnan(X), self.statistics_, X)

    def fit_transform(self, X: Array, y: Array | None = None) -> Array:
        return self.fit(X, y).transform(X)

    def _check_fitted(self) -> None:
        if self.statistics_ is None:
            raise RuntimeError("SimpleImputer is not fitted yet.")


class OneHotEncoder:
    """Incremental one-hot encoder for discrete columns.

    Categories expand as new chunks arrive. Missing values are ignored during
    fitting and encoded as all zeros during transformation.
    """

    def __init__(self, handle_unknown: str = "ignore") -> None:
        if handle_unknown != "ignore":
            raise ValueError("Only handle_unknown='ignore' is supported.")

        self.handle_unknown = handle_unknown
        self.categories_: List[List[object]] | None = None
        self.n_features_in_: int | None = None

    @staticmethod
    def _is_missing(value: object) -> bool:
        if value is None:
            return True

        try:
            return bool(np.isnan(value))
        except (TypeError, ValueError):
            return False

    def partial_fit(self, X: Array, y: Array | None = None) -> "OneHotEncoder":
        X = np.asarray(X, dtype=object)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if X.ndim != 2:
            raise ValueError("X must be a 1D or 2D array.")

        if self.n_features_in_ is None:
            self.n_features_in_ = X.shape[1]
            self.categories_ = [[] for _ in range(X.shape[1])]

        elif X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {X.shape[1]}."
            )

        for j in range(X.shape[1]):
            seen = set(self.categories_[j])

            for value in X[:, j]:
                if self._is_missing(value):
                    continue

                if value not in seen:
                    self.categories_[j].append(value)
                    seen.add(value)

        return self

    def fit(self, X: Array, y: Array | None = None) -> "OneHotEncoder":
        self.__init__(self.handle_unknown)
        return self.partial_fit(X, y)

    def transform(self, X: Array) -> Array:
        if self.categories_ is None:
            raise RuntimeError("OneHotEncoder is not fitted yet.")

        X = np.asarray(X, dtype=object)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {X.shape[1]}."
            )

        total_width = sum(len(cats) for cats in self.categories_)
        out = np.zeros((X.shape[0], total_width), dtype=float)

        offset = 0

        for j, cats in enumerate(self.categories_):
            index = {cat: k for k, cat in enumerate(cats)}

            for i, value in enumerate(X[:, j]):
                if self._is_missing(value):
                    continue

                if value in index:
                    out[i, offset + index[value]] = 1.0

            offset += len(cats)

        return out

    def fit_transform(self, X: Array, y: Array | None = None) -> Array:
        return self.fit(X, y).transform(X)