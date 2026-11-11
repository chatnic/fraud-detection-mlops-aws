"""SageMaker Processing entry point: score the held-out test period and write evaluation.json,
which the pipeline's quality gate (ConditionStep) reads.

Inputs:  /opt/ml/processing/model/model.tar.gz, /opt/ml/processing/test/data.csv
Output:  /opt/ml/processing/evaluation/evaluation.json
"""

from __future__ import annotations

import json
import tarfile
from pathlib import Path

import joblib
import pandas as pd

from fraud.evaluate import alert_metrics

BASE = Path("/opt/ml/processing")


def main() -> None:
    with tarfile.open(BASE / "model" / "model.tar.gz") as tar:
        tar.extractall(BASE / "model")
    bundle = joblib.load(BASE / "model" / "fraud_model.joblib")
    test = pd.read_csv(BASE / "test" / "data.csv")
    s = bundle["model"].predict_proba(test[bundle["features"]])[:, 1]
    m = alert_metrics(test, s, bundle["threshold"])
    report = {"metrics": {"pr_auc": {"value": m["pr_auc"]}, "recall_at_capacity": {"value": m["recall"]},
                          "precision_at_capacity": {"value": m["precision"]},
                          "dollar_recall": {"value": m["dollar_recall"]}}}
    out = BASE / "evaluation"
    out.mkdir(parents=True, exist_ok=True)
    (out / "evaluation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
