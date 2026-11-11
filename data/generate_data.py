"""Simulate 90 days of card transactions with realistic fraud patterns and late drift.

Legitimate behavior: each cardholder has a home location, typical spend level, preferred
merchant categories, active hours and a usual device.

Fraud patterns:
  card_testing      bursts of tiny online charges, then one large one
  account_takeover  new device, unusual hour, large online purchase, often electronics or gift cards
  stolen_card       in-person spending far from home in quick succession
  crypto_cashout    (appears only in the last 2 weeks) large transfers to crypto exchanges.
                    The model has never seen it, which is what the monitoring has to catch.

Usage:
    python data/generate_data.py --cardholders 8000 --days 90 --out data/raw/transactions.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

MCC = {  # category: (share of legit spend, typical amount)
    "grocery": (0.24, 55), "restaurants": (0.17, 38), "fuel": (0.10, 45), "retail": (0.13, 70),
    "online_retail": (0.12, 60), "travel": (0.03, 320), "utilities": (0.05, 110), "entertainment": (0.06, 30),
    "electronics": (0.03, 240), "gift_cards": (0.01, 60), "pharmacy": (0.05, 28), "crypto_exchange": (0.01, 150),
}
START = pd.Timestamp("2026-01-01")


def generate(cardholders: int = 8_000, days: int = 90, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cats = list(MCC)
    shares = np.array([MCC[c][0] for c in cats]); shares /= shares.sum()

    home_lat = rng.uniform(25, 48, cardholders); home_lon = rng.uniform(-123, -70, cardholders)
    spend_scale = rng.lognormal(0, 0.5, cardholders)
    rate = rng.gamma(2.0, 0.45, cardholders)                        # transactions per day
    online_pref = rng.beta(2, 4, cardholders)
    active_hour = rng.normal(14, 3, cardholders)
    tenure_days = rng.integers(30, 4000, cardholders)

    rows = []
    for cid in range(cardholders):
        n = rng.poisson(rate[cid] * days)
        if n == 0:
            continue
        t = np.sort(rng.uniform(0, days, n))
        day = np.floor(t)
        hour = rng.normal(active_hour[cid], 3.5, n) % 24
        ts = START + pd.to_timedelta(day, unit="D") + pd.to_timedelta(hour, unit="h")
        cat = rng.choice(cats, n, p=shares)
        amt = np.array([MCC[c][1] for c in cat]) * spend_scale[cid] * rng.lognormal(0, 0.6, n)
        online = (rng.random(n) < online_pref[cid]) | np.isin(cat, ["online_retail", "utilities"])
        dist = np.where(online, 0, np.abs(rng.normal(0, 8, n)) + rng.exponential(3, n) * (rng.random(n) < 0.05) * 100)
        # Legitimate travel: some cardholders spend in person far from home for a few days.
        if rng.random() < 0.12:
            t_start = rng.uniform(0, days - 5)
            on_trip = (t >= t_start) & (t < t_start + rng.uniform(2, 7)) & ~online
            dist = np.where(on_trip, rng.uniform(200, 2500), dist)
        # Legitimate small recurring charges (app subscriptions, parking) and night owls.
        tiny = rng.random(n) < 0.03
        amt = np.where(tiny, rng.uniform(0.99, 4.99, n), amt)
        if rng.random() < 0.08:
            hour = np.where(rng.random(n) < 0.4, rng.uniform(0, 5, n), hour)
            ts = START + pd.to_timedelta(day, unit="D") + pd.to_timedelta(hour, unit="h")
        rows.append(pd.DataFrame({
            "card_id": f"C{cid:06d}", "timestamp": ts, "amount": amt.round(2), "merchant_category": cat,
            "channel": np.where(online, "online", "in_person"), "distance_from_home_km": dist.round(1),
            "new_device": (online & (rng.random(n) < 0.04)).astype(int),
            "card_tenure_days": tenure_days[cid] + day.astype(int), "is_fraud": 0, "fraud_type": "",
        }))

    # --- Fraud injection ---
    def fraud_rows(kind, cid, t0):
        k = {"card_testing": rng.integers(4, 9), "account_takeover": rng.integers(1, 3),
             "stolen_card": rng.integers(2, 6), "crypto_cashout": rng.integers(1, 3)}[kind]
        gaps = np.cumsum(rng.exponential({"card_testing": 0.05, "account_takeover": 0.5,
                                          "stolen_card": 1.0, "crypto_cashout": 2.0}[kind], k))
        ts = t0 + pd.to_timedelta(gaps, unit="h")
        if kind == "card_testing":
            amt = np.r_[rng.uniform(0.5, 9, k - 1), rng.uniform(40, 900)]
            cat = np.r_[rng.choice(["online_retail", "entertainment"], k - 1), ["electronics"]]
            ch, dist, dev = "online", 0.0, rng.random(k) < 0.5
        elif kind == "account_takeover":
            amt = rng.uniform(120, 2500, k) * spend_scale[cid] ** 0.3
            cat = rng.choice(["electronics", "gift_cards", "online_retail"], k)
            ch, dist, dev = "online", 0.0, rng.random(k) < 0.7
            if rng.random() < 0.5:                                               # often in the middle of the night
                ts = ts.normalize() + pd.to_timedelta(rng.uniform(0, 5, k), unit="h")
        elif kind == "stolen_card":
            amt = rng.uniform(80, 700, k)
            cat = rng.choice(["retail", "electronics", "fuel", "gift_cards"], k)
            ch, dist, dev = "in_person", rng.uniform(20, 2500), np.zeros(k, dtype=bool)
        else:
            amt = rng.uniform(400, 3000, k) * spend_scale[cid] ** 0.2
            cat = np.array(["crypto_exchange"] * k)
            ch, dist, dev = "online", 0.0, rng.random(k) < 0.25            # mostly the cardholder's own device
        return pd.DataFrame({"card_id": f"C{cid:06d}", "timestamp": ts, "amount": np.round(amt, 2),
                             "merchant_category": cat, "channel": ch, "distance_from_home_km": np.round(dist, 1),
                             "new_device": np.asarray(dev, dtype=int),
                             "card_tenure_days": tenure_days[cid] + int((t0 - START).days),
                             "is_fraud": 1, "fraud_type": kind})

    n_incidents = int(cardholders * 0.08)
    victims = rng.choice(cardholders, n_incidents, replace=False)
    for cid in victims:
        t0 = START + pd.Timedelta(days=float(rng.uniform(1, days - 1)))
        late = t0 >= START + pd.Timedelta(days=days - 14)
        kinds, p = (["card_testing", "account_takeover", "stolen_card", "crypto_cashout"], [0.2, 0.2, 0.15, 0.45]) if late \
            else (["card_testing", "account_takeover", "stolen_card"], [0.4, 0.35, 0.25])
        rows.append(fraud_rows(rng.choice(kinds, p=p), cid, t0))

    df = pd.concat(rows, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    df.insert(0, "transaction_id", [f"TX{i:08d}" for i in range(len(df))])
    df = df[df.timestamp < START + pd.Timedelta(days=days)]
    return df


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cardholders", type=int, default=8_000)
    p.add_argument("--days", type=int, default=90)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=Path("data/raw/transactions.csv"))
    a = p.parse_args()
    df = generate(a.cardholders, a.days, a.seed)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"Wrote {len(df):,} transactions; fraud rate {df.is_fraud.mean():.2%}; "
          f"{df[df.is_fraud == 1].fraud_type.value_counts().to_dict()}")


if __name__ == "__main__":
    main()
