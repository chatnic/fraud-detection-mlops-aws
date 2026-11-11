"""Production monitoring: input drift (PSI), score drift, and outcome monitoring on
confirmed fraud labels (chargebacks), which arrive days after the transaction."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fraud.features import FEATURES

# Lifetime counters grow over time by construction, so their PSI rises even when nothing is
# wrong. They are excluded from drift monitoring to avoid permanent false alarms.
MONITORED = [f for f in FEATURES if f not in ("card_txn_count", "card_tenure_days")]


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10, categorical: bool = False) -> float:
    """Population Stability Index. Rule of thumb: < 0.1 stable, 0.1-0.25 watch, > 0.25 investigate."""
    eps = 1e-4
    if categorical:
        cats = np.union1d(np.unique(reference), np.unique(current))
        r = np.array([(reference == c).mean() for c in cats])
        c = np.array([(current == c).mean() for c in cats])
    else:
        edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
        if len(edges) < 3:
            return 0.0
        edges[0], edges[-1] = -np.inf, np.inf
        r = np.histogram(reference, edges)[0] / len(reference)
        c = np.histogram(current, edges)[0] / len(current)
    r, c = np.clip(r, eps, None), np.clip(c, eps, None)
    return float(np.sum((c - r) * np.log(c / r)))


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def weekly_report(ref: pd.DataFrame, ref_scores: np.ndarray, cur: pd.DataFrame, cur_scores: np.ndarray,
                  threshold: float, week_col: str = "week") -> pd.DataFrame:
    rows = []
    for wk, g in cur.groupby(week_col):
        s = cur_scores[g.index.to_numpy()]
        feat_psi = {f: psi(ref[f].to_numpy(), g[f].to_numpy(), categorical=(f == "merchant_category_code"))
                    for f in MONITORED}
        worst = max(feat_psi, key=feat_psi.get)
        fraud = g["is_fraud"].to_numpy() == 1
        missed = g[fraud & (s < threshold)]
        top_missed_cat = (missed.groupby("merchant_category")["amount"].sum().idxmax() if len(missed) else "")
        rows.append({
            "week": wk,
            "max_feature_psi": feat_psi[worst],
            "max_psi_feature": worst,
            "score_psi": psi(_logit(ref_scores), _logit(s)),
            "fraud_recall": float((s[fraud] >= threshold).mean()) if fraud.any() else np.nan,
            "missed_fraud_dollars": float(missed["amount"].sum()),
            "top_missed_category": top_missed_cat,
            "missed_dollars_top_category": float(missed.loc[missed.merchant_category == top_missed_cat, "amount"].sum())
            if top_missed_cat else 0.0,
        })
    return pd.DataFrame(rows)
