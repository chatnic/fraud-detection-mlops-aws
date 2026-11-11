"""Evaluation at a fixed analyst capacity: the fraud team can review a fixed share of
transactions, so what matters is how much fraud lands in that queue."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

ALERT_RATE = 0.005         # analysts can review 0.5% of transactions
REVIEW_COST = 4.0          # $ per alert reviewed


def threshold_for_alert_rate(scores: np.ndarray, rate: float = ALERT_RATE) -> float:
    return float(np.quantile(scores, 1 - rate))


def alert_metrics(df: pd.DataFrame, scores: np.ndarray, threshold: float) -> dict:
    alert = scores >= threshold
    y = df["is_fraud"].to_numpy().astype(bool)
    amt = df["amount"].to_numpy()
    tp = alert & y
    caught = float(amt[tp].sum())
    total = float(amt[y].sum())
    return {
        "transactions": int(len(df)),
        "alerts": int(alert.sum()),
        "alert_rate": float(alert.mean()),
        "precision": float(tp.sum() / max(alert.sum(), 1)),
        "recall": float(tp.sum() / max(y.sum(), 1)),
        "fraud_dollars_total": total,
        "fraud_dollars_caught": caught,
        "dollar_recall": caught / total if total else float("nan"),
        "review_cost": float(alert.sum() * REVIEW_COST),
        "net_savings": caught - float(alert.sum() * REVIEW_COST),
        "pr_auc": float(average_precision_score(y, scores)) if y.any() and len(np.unique(scores)) > 2 else float("nan"),
    }


def recall_by_type(df: pd.DataFrame, scores: np.ndarray, threshold: float) -> pd.Series:
    f = df[df["is_fraud"] == 1].assign(caught=scores[df["is_fraud"].to_numpy() == 1] >= threshold)
    return f.groupby("fraud_type")["caught"].mean()
