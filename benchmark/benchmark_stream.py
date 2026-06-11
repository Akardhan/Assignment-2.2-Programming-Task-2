"""Benchmarks for streaming NumCompute components."""

from __future__ import annotations

from pathlib import Path
import sys
from time import perf_counter

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from numcompute_stream.datasets import introduce_missing_values, iter_chunks
from numcompute_stream.ensemble import EnsembleClassifier
from numcompute_stream.io import make_classification_data, train_test_split
from numcompute_stream.pipeline import Pipeline
from numcompute_stream.preprocessing import SimpleImputer, StandardScaler
from numcompute_stream.tree import DecisionTreeClassifier


def loop_nanmean(X: np.ndarray) -> np.ndarray:
    values = []
    for j in range(X.shape[1]):
        total = 0.0
        count = 0
        for value in X[:, j]:
            if not np.isnan(value):
                total += float(value)
                count += 1
        values.append(total / count if count else np.nan)
    return np.asarray(values)


def time_call(fn, *args, repeats: int = 1, **kwargs):
    start = perf_counter()
    result = None
    for _ in range(repeats):
        result = fn(*args, **kwargs)
    return perf_counter() - start, result


def make_pipeline(model) -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="mean")),
            ("scale", StandardScaler()),
            ("model", model),
        ]
    )


def streaming_score(pipe: Pipeline, X_train, y_train, X_test, y_test, *, chunk_size: int) -> float:
    for X_chunk, y_chunk in iter_chunks(X_train, y_train, chunk_size=chunk_size):
        pipe.partial_fit(X_chunk, y_chunk)
    return pipe.score(X_test, y_test)


def main() -> None:
    X, y = make_classification_data(
        n_samples=800,
        n_features=8,
        n_classes=2,
        random_state=123,
        noise=0.55,
    )
    X = introduce_missing_values(X, missing_rate=0.05, random_state=123)

    loop_time, loop_mean = time_call(loop_nanmean, X, repeats=100)
    vector_time, vector_mean = time_call(np.nanmean, X, 0, repeats=100)

    if not np.allclose(loop_mean, vector_mean, equal_nan=True):
        raise RuntimeError("Loop and vectorised means do not match.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=17,
    )

    tree_pipe = make_pipeline(DecisionTreeClassifier(max_depth=5, random_state=17))
    forest_pipe = make_pipeline(
        EnsembleClassifier(
            n_estimators=7,
            method="random_forest",
            max_depth=5,
            random_state=17,
        )
    )

    tree_time, tree_score = time_call(
        streaming_score,
        tree_pipe,
        X_train,
        y_train,
        X_test,
        y_test,
        repeats=1,
        chunk_size=80,
    )
    forest_time, forest_score = time_call(
        streaming_score,
        forest_pipe,
        X_train,
        y_train,
        X_test,
        y_test,
        repeats=1,
        chunk_size=80,
    )

    print("Benchmark results")
    print(f"Loop nanmean:       {loop_time:.4f}s")
    print(f"Vectorised nanmean: {vector_time:.4f}s")
    print(f"Speedup:            {loop_time / max(vector_time, 1e-12):.1f}x")
    print()
    print(f"Tree stream:        accuracy={tree_score:.3f}, time={tree_time:.4f}s")
    print(f"Random forest:      accuracy={forest_score:.3f}, time={forest_time:.4f}s")


if __name__ == "__main__":
    main()
