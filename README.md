# NumCompute Stream

NumCompute Stream is a NumPy-only streaming machine learning package for Assignment 2.2. It implements incremental preprocessing, streaming statistics and metrics, decision trees, tree ensembles, pipelines, training logs, visualisation helpers, tests, demos, and benchmarks.

Only NumPy and matplotlib are used for package functionality. Pytest is used for the test suite.

Repository link: https://github.com/Akardhan/Assignment-2.2-Programming-Task-2

## How To Run

Install the required packages first:

```bash
python -m pip install numpy matplotlib pytest
```

Then run the main verification commands:

```bash
python -m pytest -q
python demo/stream_demo.py
python benchmark/benchmark_stream.py
```

Expected key output:

```text
68 passed
Chunks trained: 9
Decision tree test accuracy: 0.987
Random forest test accuracy: 1.000
Speedup: tens of times faster; about 90x on the recorded benchmark run
```

Benchmark timings vary by machine and current system load, so describe the vectorised NumPy result as approximately tens of times faster rather than relying on a single exact speedup value.

## Features

- `DecisionTreeClassifier` with Gini or entropy splits, NaN handling, depth limits, deterministic tie handling, and `partial_fit`.
- `EnsembleClassifier` with bagging or random forest style trees and streaming adaptation.
- `BoostingClassifier` with AdaBoost-style weighted resampling.
- Streaming transformers: `StandardScaler`, `SimpleImputer`, and `OneHotEncoder`.
- Streaming statistics through `StreamingStats.update_stats`.
- Streaming classification metrics with `update`, `result`, and `reset`.
- `Pipeline.partial_fit` for chunk-wise transformer and model updates.
- `StreamTrainer` for per-chunk logs, memory estimates, chunk accuracy, cumulative accuracy, and rolling accuracy.
- Matplotlib visualisation functions in `numcompute_stream.visualise`.

## Project Layout

```text
numcompute_stream/      Core package
tests/                  Unit tests
demo/                   Streaming demo script and notebook
benchmark/              Benchmark script
README.md               Usage instructions
```

## Requirement Mapping

- Streaming learning: `numcompute_stream/stream.py`, `numcompute_stream/pipeline.py`, `numcompute_stream/tree.py`, `numcompute_stream/ensemble.py`
- Decision tree: `numcompute_stream/tree.py`
- Ensemble model: `numcompute_stream/ensemble.py`
- Preprocessing: `numcompute_stream/preprocessing.py`
- Streaming statistics: `numcompute_stream/stats.py`
- Metrics: `numcompute_stream/metrics.py`
- Visualisation: `numcompute_stream/visualise.py`
- Demo notebook: `demo/stream_demo.ipynb`
- Benchmark: `benchmark/benchmark_stream.py`
- Tests: `tests/`

## Setup

Use Python 3.10 or newer.

```bash
python -m pip install numpy matplotlib pytest
```

## Run Tests

```bash
python -m pytest -q
```

The current suite contains 68 tests covering preprocessing, statistics, metrics, trees, ensembles, pipelines, visualisation, stream training, and edge cases.

## Run The Demo

The demo generates a numeric classification dataset, saves it as CSV, reloads it through `numcompute_stream.io.load_csv`, splits it into stream chunks, trains a single tree and random forest incrementally, and writes plots to `demo/figures/`.

```bash
python demo/stream_demo.py
```

You can also open `demo/stream_demo.ipynb` for the notebook version required by the assignment.

## Run Benchmarks

```bash
python benchmark/benchmark_stream.py
```

The benchmark reports:

- loop-based versus NumPy-vectorised NaN-safe mean timing
- streaming single-tree accuracy and runtime
- streaming random-forest accuracy and runtime

In the recorded benchmark run, the vectorised NumPy version was about 90x faster than the loop-based version.

## Minimal Usage

```python
from numcompute_stream.preprocessing import SimpleImputer, StandardScaler
from numcompute_stream.tree import DecisionTreeClassifier
from numcompute_stream.pipeline import Pipeline
from numcompute_stream.stream import StreamTrainer

pipe = Pipeline([
    ("impute", SimpleImputer(strategy="mean")),
    ("scale", StandardScaler()),
    ("model", DecisionTreeClassifier(max_depth=4, random_state=7)),
])

trainer = StreamTrainer(pipe)

for X_chunk, y_chunk in stream_chunks:
    log_row = trainer.fit_chunk(X_chunk, y_chunk)
    print(log_row["chunk"], log_row["chunk_accuracy"])
```

## Submission Checklist

- Submit the GitHub repository link: https://github.com/Akardhan/Assignment-2.2-Programming-Task-2
- Include the `demo/stream_demo.ipynb` notebook.
- Record a demo video of setup, demo execution, visualisations, predictions, and logs.
- Prepare the PDF report with design decisions, testing and edge cases, benchmark results, and reflections.
- Follow the course AI-use acknowledgement rules if AI assistance was used.
