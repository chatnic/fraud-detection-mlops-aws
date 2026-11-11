"""SageMaker Processing entry point: raw transactions -> point-in-time features -> time-based splits.

Inputs:  /opt/ml/processing/input/transactions.csv
Outputs: /opt/ml/processing/{train,validation,test}/data.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from fraud.features import build_features

BASE = Path("/opt/ml/processing")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train-days", type=int, default=56)
    p.add_argument("--validation-days", type=int, default=13)
    a = p.parse_args()

    raw = pd.read_csv(BASE / "input" / "transactions.csv", parse_dates=["timestamp"])
    df = build_features(raw)
    day = (df["timestamp"] - df["timestamp"].min().normalize()).dt.days
    splits = {
        "train": df[day < a.train_days],
        "validation": df[(day >= a.train_days) & (day < a.train_days + a.validation_days)],
        "test": df[day >= a.train_days + a.validation_days],
    }
    for name, part in splits.items():
        out = BASE / name
        out.mkdir(parents=True, exist_ok=True)
        part.to_csv(out / "data.csv", index=False)
        print(f"{name}: {len(part):,} rows, fraud rate {part.is_fraud.mean():.3%}")


if __name__ == "__main__":
    main()
