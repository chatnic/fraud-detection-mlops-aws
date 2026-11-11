"""Point-in-time behavioral features.

Every feature for a transaction uses only that cardholder's *earlier* transactions, the
same information a real-time scoring service would have when the card is swiped. In
production these come from an online feature store (for example SageMaker Feature Store);
here they are computed in batch with the same logic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CATEGORIES = ["grocery", "restaurants", "fuel", "retail", "online_retail", "travel", "utilities",
              "entertainment", "electronics", "gift_cards", "pharmacy", "crypto_exchange"]

FEATURES = [
    "amount", "log_amount", "hour", "is_night", "is_online", "new_device", "distance_from_home_km",
    "card_tenure_days", "merchant_category_code",
    "txn_count_1h", "txn_count_24h", "amount_sum_24h", "small_txn_count_1h",
    "hours_since_prev_txn", "amount_vs_card_avg", "card_avg_amount", "card_txn_count",
    "first_time_category", "online_share_card",
]
CATEGORICAL_IDX = [FEATURES.index("merchant_category_code")]
# Prior used for a card with no history. A fixed constant: a dataset-wide statistic here would
# leak future transactions into past features (the point-in-time test caught exactly that).
DEFAULT_AVG_AMOUNT = 75.0


def _rolling_prior(g: pd.DataFrame, window: str, col: str, how: str) -> pd.Series:
    r = g.rolling(window, on="timestamp", closed="left")[col]
    return (r.count() if how == "count" else r.sum()).fillna(0)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["card_id", "timestamp", "transaction_id"]).reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    out = df.copy()
    out["log_amount"] = np.log1p(out["amount"])
    out["hour"] = out["timestamp"].dt.hour + out["timestamp"].dt.minute / 60
    out["is_night"] = ((out["hour"] < 5) | (out["hour"] >= 23)).astype(int)
    out["is_online"] = (out["channel"] == "online").astype(int)
    out["merchant_category_code"] = out["merchant_category"].map({c: i for i, c in enumerate(CATEGORIES)}).fillna(-1)

    out["small"] = (out["amount"] < 5).astype(int)
    parts = []
    for _, g in out[["card_id", "timestamp", "amount", "small"]].groupby("card_id", sort=False):
        g = g.copy()
        g["txn_count_1h"] = _rolling_prior(g, "1h", "amount", "count")
        g["txn_count_24h"] = _rolling_prior(g, "24h", "amount", "count")
        g["amount_sum_24h"] = _rolling_prior(g, "24h", "amount", "sum")
        g["small_txn_count_1h"] = _rolling_prior(g, "1h", "small", "sum")
        parts.append(g[["txn_count_1h", "txn_count_24h", "amount_sum_24h", "small_txn_count_1h"]])
    out = out.join(pd.concat(parts))

    grp = out.groupby("card_id", sort=False)
    out["hours_since_prev_txn"] = grp["timestamp"].diff().dt.total_seconds().div(3600).fillna(24 * 30)
    prior_sum = grp["amount"].cumsum() - out["amount"]
    out["card_txn_count"] = grp.cumcount()
    out["card_avg_amount"] = (prior_sum / out["card_txn_count"].replace(0, np.nan)).fillna(DEFAULT_AVG_AMOUNT)
    out["amount_vs_card_avg"] = out["amount"] / out["card_avg_amount"].clip(lower=1)
    seen_before = out.groupby(["card_id", "merchant_category"], sort=False).cumcount() > 0
    out["first_time_category"] = ((~seen_before) & (out["card_txn_count"] >= 10)).astype(int)
    prior_online = grp["is_online"].cumsum() - out["is_online"]
    out["online_share_card"] = (prior_online / out["card_txn_count"].replace(0, np.nan)).fillna(0.3)
    return out.drop(columns=["small"]).sort_values(["timestamp", "transaction_id"]).reset_index(drop=True)
