"""Small NumPy-only I/O and streaming helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional, Tuple

import numpy as np


Array = np.ndarray


def load_csv(
    path: str | Path,
    *,
    delimiter: str = ',',
    skip_header: bool = True,
    target_col: int = -1,
    dtype: type = float,
) -> Tuple[Array, Array]:
    """Load a numeric CSV file and split it into features and target.

    Parameters
  
    path:
        CSV file path.
    delimiter:
        Column delimiter.
    skip_header:
        Whether to skip the first row.
    target_col:
        Index of the target column. Defaults to the last column.
    dtype:
        Numeric dtype used by ``numpy.genfromtxt``.

    Returns
  
    X, y:
        Feature matrix of shape ``(n_samples, n_features)`` and target vector.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'CSV file not found: {path}')
    data = np.genfromtxt(
        path,
        delimiter=delimiter,
        skip_header=1 if skip_header else 0,
        dtype=dtype,
    )
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 2:
        raise ValueError('CSV must contain at least one feature column and one target column.')
    target_col = target_col if target_col >= 0 else data.shape[1] + target_col
    if target_col < 0 or target_col >= data.shape[1]:
        raise ValueError('target_col is outside the valid column range.')
    y = data[:, target_col]
    X = np.delete(data, target_col, axis=1)
    return X, y


def save_csv(path: str | Path, X: Array, y: Array, *, header: Optional[str] = None) -> None:
    """Save features and target to CSV using NumPy."""
    X = ensure_2d(X)
    y = ensure_1d(y)
    if X.shape[0] != y.shape[0]:
        raise ValueError('X and y must have the same number of samples.')
    data = np.column_stack([X, y])
    np.savetxt(path, data, delimiter=',', header=header or '', comments='')


def train_test_split(
    X: Array,
    y: Array,
    *,
    test_size: float = 0.2,
    shuffle: bool = True,
    random_state: Optional[int] = None,
) -> Tuple[Array, Array, Array, Array]:
    """Split arrays into train and test sets without using scikit-learn."""
    X = ensure_2d(X)
    y = ensure_1d(y)
    if X.shape[0] != y.shape[0]:
        raise ValueError('X and y must have the same number of samples.')
    if not 0 < test_size < 1:
        raise ValueError('test_size must be between 0 and 1.')
    n = X.shape[0]
    rng = np.random.default_rng(random_state)
    idx = np.arange(n)
    if shuffle:
        rng.shuffle(idx)
    test_n = max(1, int(round(n * test_size)))
    test_idx = idx[:test_n]
    train_idx = idx[test_n:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def chunk_iterator(X: Array, y: Optional[Array] = None, *, chunk_size: int = 32) -> Iterator:
    """Yield data in fixed-size chunks.

    If ``y`` is provided, each yielded item is ``(X_chunk, y_chunk)``.
    Otherwise only ``X_chunk`` is yielded.
    """
    X = ensure_2d(X)
    if chunk_size <= 0:
        raise ValueError('chunk_size must be positive.')
    if y is not None:
        y = ensure_1d(y)
        if X.shape[0] != y.shape[0]:
            raise ValueError('X and y must have the same number of samples.')
    for start in range(0, X.shape[0], chunk_size):
        end = min(start + chunk_size, X.shape[0])
        if y is None:
            yield X[start:end]
        else:
            yield X[start:end], y[start:end]


def ensure_2d(X: Array) -> Array:
    """Return X as a two-dimensional float array."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.ndim != 2:
        raise ValueError('X must be a 1D or 2D array.')
    return X


def ensure_1d(y: Array) -> Array:
    """Return y as a one-dimensional array."""
    y = np.asarray(y)
    if y.ndim != 1:
        y = y.ravel()
    return y


def make_classification_data(
    n_samples: int = 600,
    n_features: int = 5,
    n_classes: int = 2,
    *,
    random_state: Optional[int] = 42,
    noise: float = 0.35,
) -> Tuple[Array, Array]:
    """Generate a simple numeric classification dataset with NumPy only.

    The data is created from class-specific Gaussian clusters plus noise.
    """
    if n_samples <= 0 or n_features <= 0 or n_classes <= 1:
        raise ValueError('n_samples and n_features must be positive; n_classes must be > 1.')
    rng = np.random.default_rng(random_state)
    samples_per_class = np.full(n_classes, n_samples // n_classes, dtype=int)
    samples_per_class[: n_samples % n_classes] += 1
    centers = rng.normal(0, 3, size=(n_classes, n_features))
    X_parts = []
    y_parts = []
    for class_id, count in enumerate(samples_per_class):
        X_class = centers[class_id] + rng.normal(0, 1 + noise, size=(count, n_features))
        X_parts.append(X_class)
        y_parts.append(np.full(count, class_id))
    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)
    idx = rng.permutation(n_samples)
    return X[idx], y[idx]
