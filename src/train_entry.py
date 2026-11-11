"""SageMaker training entry point (script mode, SKLearn framework container).

SageMaker passes data channels and the model directory through environment variables.
Running locally works too:
    SM_CHANNEL_TRAIN=... SM_CHANNEL_VALIDATION=... SM_MODEL_DIR=models python src/train_entry.py
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import joblib
import pandas as pd

from fraud.evaluate import ALERT_RATE, threshold_for_alert_rate
from fraud.features import FEATURES
from fraud.model import score, train


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", default=os.environ.get("SM_MODEL_DIR", "models"))
    p.add_argument("--train", default=os.environ.get("SM_CHANNEL_TRAIN", "data/processed/train"))
    p.add_argument("--validation", default=os.environ.get("SM_CHANNEL_VALIDATION", "data/processed/validation"))
    p.add_argument("--alert-rate", type=float, default=ALERT_RATE)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    tr = pd.read_csv(Path(a.train) / "data.csv")
    va = pd.read_csv(Path(a.validation) / "data.csv")
    model = train(tr, a.seed)
    thr = threshold_for_alert_rate(score(model, va), a.alert_rate)

    Path(a.model_dir).mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "threshold": thr, "features": FEATURES}, Path(a.model_dir) / "fraud_model.joblib")
    (Path(a.model_dir) / "model_meta.json").write_text(json.dumps({"threshold": thr, "alert_rate": a.alert_rate,
                                                                    "n_train": len(tr)}))
    print(f"Trained on {len(tr):,} rows; alert threshold {thr:.4f} for {a.alert_rate:.1%} alert rate")


if __name__ == "__main__":
    main()
