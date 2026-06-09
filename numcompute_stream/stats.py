"""Streaming statistics utilities."""

from __future__ import annotations

from typing import Iterable, Optional, Tuple

import numpy as np


Array = np.ndarray


def _ensure_2d(X: Array) -> Array:
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.ndim != 2:
        raise ValueError('X must be 1D or 2D.')
    return X


def nanmean(X: Array, axis: int | None = 0) -> Array:
    """NaN-safe mean wrapper with consistent error handling."""
    return np.nanmean(np.asarray(X, dtype=float), axis=axis)


def nanvariance(X: Array, axis: int | None = 0, ddof: int = 0) -> Array:
    """NaN-safe variance wrapper."""
    return np.nanvar(np.asarray(X, dtype=float), axis=axis, ddof=ddof)


class RunningStats:
    """Track streaming mean, variance, min, max, quantiles and histograms.

    Mean/variance use a vectorised parallel Welford update. Quantiles are
    estimated from a bounded reservoir of recent observations to keep memory
    usage predictable.
    """

    def __init__(self, *, reservoir_size: int = 10000, random_state: Optional[int] = 42) -> None:
        if reservoir_size <= 0:
            raise ValueError('reservoir_size must be positive.')
        self.reservoir_size = reservoir_size
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)
        self.n_features_in_: int | None = None
        self.n_seen_: np.ndarray | None = None
        self.mean_: np.ndarray | None = None
        self.M2_: np.ndarray | None = None
        self.min_: np.ndarray | None = None
        self.max_: np.ndarray | None = None
        self.reservoir_: np.ndarray | None = None
        self.total_rows_: int = 0

    def update_stats(self, X_chunk: Array) -> 'RunningStats':
        X = _ensure_2d(X_chunk)
        if self.n_features_in_ is None:
            self.n_features_in_ = X.shape[1]
            self.n_seen_ = np.zeros(X.shape[1], dtype=float)
            self.mean_ = np.zeros(X.shape[1], dtype=float)
            self.M2_ = np.zeros(X.shape[1], dtype=float)
            self.min_ = np.full(X.shape[1], np.inf)
            self.max_ = np.full(X.shape[1], -np.inf)
            self.reservoir_ = np.empty((0, X.shape[1]), dtype=float)
        elif X.shape[1] != self.n_features_in_:
            raise ValueError(f'Expected {self.n_features_in_} features, got {X.shape[1]}.')

        mask = ~np.isnan(X)
        counts = mask.sum(axis=0).astype(float)
        sums = np.where(mask, X, 0.0).sum(axis=0)
        chunk_mean = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
        centered = np.where(mask, X - chunk_mean, 0.0)
        chunk_M2 = np.sum(centered * centered, axis=0)
        total = self.n_seen_ + counts
        delta = chunk_mean - self.mean_
        valid = counts > 0
        self.mean_[valid] += delta[valid] * counts[valid] / total[valid]
        self.M2_[valid] += chunk_M2[valid] + (delta[valid] ** 2) * self.n_seen_[valid] * counts[valid] / total[valid]
        self.n_seen_[valid] = total[valid]
        with np.errstate(all='ignore'):
            self.min_ = np.fmin(self.min_, np.nanmin(X, axis=0))
            self.max_ = np.fmax(self.max_, np.nanmax(X, axis=0))
        self._update_reservoir(X)
        self.total_rows_ += X.shape[0]
        return self

    def _update_reservoir(self, X: Array) -> None:
        if self.reservoir_ is None:
            self.reservoir_ = X.copy()
            return
        combined = np.vstack([self.reservoir_, X])
        if combined.shape[0] <= self.reservoir_size:
            self.reservoir_ = combined
            return
        idx = self.rng.choice(combined.shape[0], size=self.reservoir_size, replace=False)
        self.reservoir_ = combined[idx]

    def mean(self) -> Array:
        self._check_fitted()
        return self.mean_.copy()

    def variance(self, ddof: int = 0) -> Array:
        self._check_fitted()
        denom = np.maximum(self.n_seen_ - ddof, 1)
        return self.M2_ / denom

    def std(self, ddof: int = 0) -> Array:
        return np.sqrt(self.variance(ddof=ddof))

    def min(self) -> Array:
        self._check_fitted()
        return self.min_.copy()

    def max(self) -> Array:
        self._check_fitted()
        return self.max_.copy()

    def quantile(self, q: float | Iterable[float]) -> Array:
        self._check_fitted()
        if self.reservoir_.size == 0:
            raise RuntimeError('No reservoir data available.')
        return np.nanquantile(self.reservoir_, q, axis=0)

    def histogram(self, feature_index: int = 0, bins: int = 10, range: Optional[Tuple[float, float]] = None):
        self._check_fitted()
        if feature_index < 0 or feature_index >= self.n_features_in_:
            raise ValueError('feature_index is out of range.')
        values = self.reservoir_[:, feature_index]
        values = values[~np.isnan(values)]
        return np.histogram(values, bins=bins, range=range)

    def reset(self) -> None:
        self.__init__(reservoir_size=self.reservoir_size, random_state=self.random_state)

    def _check_fitted(self) -> None:
        if self.mean_ is None:
            raise RuntimeError('RunningStats has not received any data yet.')
