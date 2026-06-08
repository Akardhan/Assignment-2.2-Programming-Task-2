from __future__ import annotations

from typing import Iterator, Optional, Tuple

import numpy as np


Array = np.ndarray


def make_classification_stream(
    n_samples: int = 500,
    n_features: int = 6,
    n_informative: Optional[int] = None,
    n_classes: int = 2,
    class_sep: float = 2.0,
    noise: float = 0.6,
    missing_rate: float = 0.0,
    random_state: Optional[int] = None,
) -> tuple[Array, Array]:
   

    if n_samples <= 0:
        raise ValueError('n_samples must be positive.')
    if n_features <= 0:
        raise ValueError('n_features must be positive.')
    if n_classes <= 1:
        raise ValueError('n_classes must be at least 2.')
    if n_informative is None:
        n_informative = max(1, min(n_features, n_classes + 1))
    if not 1 <= n_informative <= n_features:
        raise ValueError('n_informative must be between 1 and n_features.')
    if not 0 <= missing_rate < 1:
        raise ValueError('missing_rate must be in [0, 1).')

    rng = np.random.default_rng(random_state)
    y = rng.integers(0, n_classes, size=n_samples)
    centroids = rng.normal(0.0, class_sep, size=(n_classes, n_informative))
    informative = centroids[y] + rng.normal(0.0, noise, size=(n_samples, n_informative))
    if n_features > n_informative:
        redundant = rng.normal(0.0, 1.0, size=(n_samples, n_features - n_informative))
        X = np.hstack([informative, redundant])
    else:
        X = informative
    order = rng.permutation(n_samples)
    X = X[order].astype(float)
    y = y[order]
    if missing_rate > 0:
        X = introduce_missing_values(X, missing_rate=missing_rate, random_state=random_state)
    return X, y


def introduce_missing_values(X: Array, missing_rate: float = 0.05, random_state: Optional[int] = None) -> Array:
    """Return a copy of ``X`` with randomly inserted NaN values."""

    if not 0 <= missing_rate < 1:
        raise ValueError('missing_rate must be in [0, 1).')
    rng = np.random.default_rng(random_state)
    X = np.asarray(X, dtype=float).copy()
    mask = rng.random(X.shape) < missing_rate
    X[mask] = np.nan
    return X


def iter_chunks(
    X: Array,
    y: Optional[Array] = None,
    *,
    chunk_size: Optional[int] = None,
    n_chunks: Optional[int] = None,
    shuffle: bool = False,
    random_state: Optional[int] = None,
) -> Iterator[Tuple[Array, Optional[Array]]]:
    """Yield fixed-size or approximately equal chunks from arrays."""

    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    y_arr = None if y is None else np.asarray(y).ravel()
    if y_arr is not None and X.shape[0] != y_arr.shape[0]:
        raise ValueError('X and y must contain the same number of samples.')
    if chunk_size is None and n_chunks is None:
        raise ValueError('Provide either chunk_size or n_chunks.')
    if chunk_size is not None and chunk_size <= 0:
        raise ValueError('chunk_size must be positive.')
    if n_chunks is not None and n_chunks <= 0:
        raise ValueError('n_chunks must be positive.')

    idx = np.arange(X.shape[0])
    if shuffle:
        rng = np.random.default_rng(random_state)
        rng.shuffle(idx)
    if n_chunks is not None:
        splits = np.array_split(idx, n_chunks)
    else:
        splits = [idx[start:start + chunk_size] for start in range(0, len(idx), chunk_size)]
    for part in splits:
        if part.size == 0:
            continue
        yield X[part], None if y_arr is None else y_arr[part]


def train_test_split(
    X: Array,
    y: Array,
    *,
    test_size: float = 0.25,
    random_state: Optional[int] = None,
    shuffle: bool = True,
) -> tuple[Array, Array, Array, Array]:
    """NumPy-only train/test split helper."""

    if not 0 < test_size < 1:
        raise ValueError('test_size must be in (0, 1).')
    X = np.asarray(X)
    y = np.asarray(y).ravel()
    if X.shape[0] != y.shape[0]:
        raise ValueError('X and y must contain the same number of samples.')
    idx = np.arange(X.shape[0])
    if shuffle:
        rng = np.random.default_rng(random_state)
        rng.shuffle(idx)
    n_test = max(1, int(round(X.shape[0] * test_size)))
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    if train_idx.size == 0:
        raise ValueError('test_size leaves no training rows.')
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]
