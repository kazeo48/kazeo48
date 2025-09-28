"""Utility functions for generating and modeling math tutoring enrollment data."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


@dataclass
class DatasetConfig:
    """Configuration options for synthetic dataset generation."""

    random_state: int = 42
    grade_levels: Tuple[int, ...] = tuple(range(7, 13))
    learning_styles: Tuple[str, ...] = ("visual", "auditory", "kinesthetic", "reading")


def _simulate_base_features(n_samples: int, config: DatasetConfig) -> pd.DataFrame:
    rng = np.random.default_rng(config.random_state)

    grade = rng.choice(config.grade_levels, size=n_samples)
    math_score = rng.normal(loc=75, scale=10, size=n_samples).clip(0, 100)
    hours_studied = rng.normal(loc=5, scale=2, size=n_samples).clip(0, None)
    parent_support = rng.uniform(0, 1, size=n_samples)
    extracurricular_load = rng.integers(0, 5, size=n_samples)
    learning_style = rng.choice(config.learning_styles, size=n_samples)

    return pd.DataFrame(
        {
            "grade": grade,
            "math_score": math_score,
            "hours_studied": hours_studied,
            "parent_support": parent_support,
            "extracurricular_load": extracurricular_load,
            "learning_style": learning_style,
        }
    )


def _calculate_tutoring_probability(df: pd.DataFrame) -> np.ndarray:
    """Calculate tutoring propensity based on hand-crafted heuristics."""

    base = 0.2
    low_score_influence = np.where(df["math_score"] < 70, 0.3, -0.05)
    grade_influence = (12 - df["grade"]) * 0.01
    parent_support_influence = (1 - df["parent_support"]) * 0.2
    workload_influence = df["extracurricular_load"] * -0.03
    hours_influence = np.where(df["hours_studied"] < 4, 0.05, -0.02)

    logits = base + (
        low_score_influence
        + grade_influence
        + parent_support_influence
        + workload_influence
        + hours_influence
    )

    return 1 / (1 + np.exp(-logits))


def generate_math_tutoring_dataset(
    n_samples: int, config: DatasetConfig | None = None
) -> pd.DataFrame:
    """Generate a synthetic dataset describing math tutoring enrollment."""

    if config is None:
        config = DatasetConfig()

    features = _simulate_base_features(n_samples, config)
    propensity = _calculate_tutoring_probability(features)

    rng = np.random.default_rng(config.random_state)
    tutoring = rng.binomial(1, propensity)

    features["takes_math_tutoring"] = tutoring
    return features


def build_model_pipeline() -> Pipeline:
    """Create a preprocessing + logistic regression pipeline."""

    categorical_features = ["learning_style"]
    numeric_features = [
        "grade",
        "math_score",
        "hours_studied",
        "parent_support",
        "extracurricular_load",
    ]

    preprocess = ColumnTransformer(
        transformers=[
            (
                "learning_style",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features,
            ),
            ("numeric", "passthrough", numeric_features),
        ]
    )

    model = LogisticRegression(max_iter=1000)

    return Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", model),
        ]
    )


def train_test_split_dataset(
    df: pd.DataFrame,
    test_size: float = 0.1,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split the dataset into train/test partitions."""

    X = df.drop(columns=["takes_math_tutoring"])
    y = df["takes_math_tutoring"]
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
