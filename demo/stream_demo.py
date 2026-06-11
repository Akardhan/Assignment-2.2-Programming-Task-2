"""Runnable streaming demo for Assignment 2.2.

The demo is self-contained: it creates a small numeric CSV dataset, loads that
CSV through the custom I/O module, trains a single decision tree and a random
forest style ensemble chunk by chunk, and saves visualisations to demo/figures.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MPLCONFIGDIR = PROJECT_ROOT / "demo" / ".matplotlib-cache"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

CACHE_DIR = PROJECT_ROOT / "demo" / ".cache"
FONTCONFIG_CACHE = CACHE_DIR / "fontconfig"
FONTCONFIG_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))
os.environ.setdefault("FC_CACHEDIR", str(FONTCONFIG_CACHE))

import matplotlib

matplotlib.use("Agg")

from numcompute_stream.datasets import iter_chunks
from numcompute_stream.ensemble import EnsembleClassifier
from numcompute_stream.io import load_csv, make_classification_data, save_csv, train_test_split
from numcompute_stream.pipeline import Pipeline
from numcompute_stream.preprocessing import SimpleImputer, StandardScaler
from numcompute_stream.stream import StreamTrainer
from numcompute_stream.tree import DecisionTreeClassifier
from numcompute_stream.visualise import (
    compare_models,
    plot_metric_over_time,
    plot_predictions_vs_ground_truth,
)


DATA_PATH = PROJECT_ROOT / "demo" / "stream_demo_data.csv"
FIGURE_DIR = PROJECT_ROOT / "demo" / "figures"


def ensure_demo_csv() -> Path:
    """Create the demo CSV if it does not already exist."""

    if DATA_PATH.exists():
        return DATA_PATH

    X, y = make_classification_data(
        n_samples=600,
        n_features=6,
        n_classes=2,
        random_state=42,
        noise=0.45,
    )
    header = ",".join([f"feature_{i}" for i in range(X.shape[1])] + ["target"])
    save_csv(DATA_PATH, X, y, header=header)
    return DATA_PATH


def make_tree_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="mean")),
            ("scale", StandardScaler()),
            ("model", DecisionTreeClassifier(max_depth=5, random_state=7)),
        ]
    )


def make_ensemble_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="mean")),
            ("scale", StandardScaler()),
            (
                "model",
                EnsembleClassifier(
                    n_estimators=7,
                    method="random_forest",
                    max_depth=5,
                    random_state=7,
                ),
            ),
        ]
    )


def train_stream(pipe: Pipeline, X_train, y_train, *, chunk_size: int) -> StreamTrainer:
    trainer = StreamTrainer(pipe)
    for X_chunk, y_chunk in iter_chunks(X_train, y_train, chunk_size=chunk_size):
        trainer.fit_chunk(X_chunk, y_chunk)
    return trainer


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = ensure_demo_csv()
    X, y = load_csv(csv_path, skip_header=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=11,
    )

    tree_trainer = train_stream(make_tree_pipeline(), X_train, y_train, chunk_size=50)
    ensemble_trainer = train_stream(make_ensemble_pipeline(), X_train, y_train, chunk_size=50)

    tree_score = tree_trainer.model.score(X_test, y_test)
    ensemble_score = ensemble_trainer.model.score(X_test, y_test)
    y_pred = ensemble_trainer.model.predict(X_test)

    tree_accuracy = tree_trainer.metric_values("chunk_accuracy")
    ensemble_accuracy = ensemble_trainer.metric_values("chunk_accuracy")

    plot_metric_over_time(
        ensemble_accuracy,
        title="Streaming ensemble accuracy",
        ylabel="Chunk accuracy",
        save_path=FIGURE_DIR / "ensemble_accuracy.png",
        show=False,
    )
    compare_models(
        tree_accuracy,
        ensemble_accuracy,
        labels=("Decision tree", "Random forest"),
        title="Streaming model comparison",
        ylabel="Chunk accuracy",
        save_path=FIGURE_DIR / "model_comparison.png",
        show=False,
    )
    plot_predictions_vs_ground_truth(
        y_test[:80],
        y_pred[:80],
        title="Predictions vs ground truth",
        save_path=FIGURE_DIR / "predictions_vs_ground_truth.png",
        show=False,
    )

    print("Stream demo complete")
    print(f"Loaded CSV: {csv_path.relative_to(PROJECT_ROOT)}")
    print(f"Chunks trained: {len(ensemble_trainer.logs_)}")
    print(f"Decision tree test accuracy: {tree_score:.3f}")
    print(f"Random forest test accuracy: {ensemble_score:.3f}")
    print(f"Figures written to: {FIGURE_DIR.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
