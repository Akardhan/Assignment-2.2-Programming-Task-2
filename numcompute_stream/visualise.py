"""Reusable matplotlib visualisation functions for streaming logs."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt


def _finish_plot(save_path: Optional[str | Path] = None, show: bool = True):
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    if show:
        plt.show()
    fig = plt.gcf()
    return fig


def plot_metric_over_time(metric_values: Iterable[float], title: str = 'Metric over time', ylabel: str = 'Metric', *, save_path=None, show: bool = True):
    """Plot a metric, such as accuracy, across stream chunks."""
    values = np.asarray(list(metric_values), dtype=float)
    plt.figure()
    plt.plot(np.arange(len(values)), values, marker='o')
    plt.title(title)
    plt.xlabel('Chunk')
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    return _finish_plot(save_path, show)


def compare_models(metric1: Sequence[float], metric2: Sequence[float], labels=('Model 1', 'Model 2'), *, title='Model comparison', ylabel='Metric', save_path=None, show: bool = True):
    """Compare two model metric histories."""
    m1 = np.asarray(metric1, dtype=float)
    m2 = np.asarray(metric2, dtype=float)
    plt.figure()
    plt.plot(np.arange(len(m1)), m1, marker='o', label=labels[0])
    plt.plot(np.arange(len(m2)), m2, marker='s', label=labels[1])
    plt.title(title)
    plt.xlabel('Chunk')
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, alpha=0.3)
    return _finish_plot(save_path, show)


def plot_predictions_vs_ground_truth(y_true, y_pred, *, title='Predictions vs ground truth', save_path=None, show: bool = True):
    """Visualise predicted labels against true labels for the latest chunk."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError('y_true and y_pred must have the same length.')
    x = np.arange(y_true.shape[0])
    plt.figure()
    plt.scatter(x, y_true, label='Ground truth', marker='o')
    plt.scatter(x, y_pred, label='Prediction', marker='x')
    plt.title(title)
    plt.xlabel('Sample index')
    plt.ylabel('Class label')
    plt.legend()
    plt.grid(True, alpha=0.3)
    return _finish_plot(save_path, show)
