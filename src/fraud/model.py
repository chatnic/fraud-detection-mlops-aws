"""Fraud model and the rule baseline it replaces."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from fraud.features import CATEGORICAL_IDX, FEATURES


def make_model(seed: int = 42) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=600, max_leaf_nodes=31, min_samples_leaf=50,
        l2_regularization=1.0, class_weight={0: 1, 1: 5}, categorical_features=CATEGORICAL_IDX,
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=30, scoring="average_precision",
        random_state=seed,
    )


def train(df: pd.DataFrame, seed: int = 42) -> HistGradientBoostingClassifier:
    return make_model(seed).fit(df[FEATURES], df["is_fraud"])


def score(model, df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(df[FEATURES])[:, 1]


def rule_score(df: pd.DataFrame) -> np.ndarray:
    """A typical legacy rule set: any rule firing raises an alert (score 1)."""
    rules = (
        (df["amount"] > 1000)
        | ((df["distance_from_home_km"] > 500) & (df["is_online"] == 0))
        | ((df["new_device"] == 1) & (df["amount"] > 300))
        | (df["txn_count_1h"] >= 5)
    )
    return rules.astype(float).to_numpy()
