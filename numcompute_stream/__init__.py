"""
NumCompute Stream

A lightweight NumPy-based streaming machine learning framework with
incremental preprocessing, statistics, metrics, decision trees, ensembles,
pipelines, and visualisation utilities.
"""

from .preprocessing import StandardScaler, SimpleImputer, OneHotEncoder
from .stats import StreamingStats
from .metrics import (
    StreamingAccuracy,
    StreamingPrecision,
    StreamingRecall,
    StreamingF1,
    StreamingConfusionMatrix,
    StreamingAUC,
)
from .tree import DecisionTreeClassifier
from .ensemble import EnsembleClassifier, BoostingClassifier
from .pipeline import Pipeline
from .stream import StreamTrainer

__all__ = [
    "StandardScaler",
    "SimpleImputer",
    "OneHotEncoder",
    "StreamingStats",
    "StreamingAccuracy",
    "StreamingPrecision",
    "StreamingRecall",
    "StreamingF1",
    "StreamingConfusionMatrix",
    "StreamingAUC",
    "DecisionTreeClassifier",
    "EnsembleClassifier",
    "BoostingClassifier",
    "Pipeline",
    "StreamTrainer",
]