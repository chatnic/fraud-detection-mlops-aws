"""SageMaker real-time inference handlers (SKLearn container).

Request (application/json): {"instances": [{<feature>: value, ...}, ...]}
Response: {"predictions": [{"fraud_score": 0.93, "alert": true}, ...]}

Features must be computed point-in-time by the caller, usually from an online feature
store, with the same logic as fraud/features.py.
"""

from __future__ import annotations

import json
import os

import joblib
import pandas as pd


def model_fn(model_dir: str):
    return joblib.load(os.path.join(model_dir, "fraud_model.joblib"))


def input_fn(body, content_type: str = "application/json") -> pd.DataFrame:
    if content_type == "application/json":
        payload = json.loads(body)
        return pd.DataFrame(payload["instances"] if isinstance(payload, dict) else payload)
    if content_type == "text/csv":
        from io import StringIO
        return pd.read_csv(StringIO(body if isinstance(body, str) else body.decode()))
    raise ValueError(f"Unsupported content type: {content_type}")


def predict_fn(df: pd.DataFrame, bundle) -> list[dict]:
    missing = [f for f in bundle["features"] if f not in df.columns]
    if missing:
        raise ValueError(f"Missing features: {missing}")
    s = bundle["model"].predict_proba(df[bundle["features"]])[:, 1]
    return [{"fraud_score": float(v), "alert": bool(v >= bundle["threshold"])} for v in s]


def output_fn(predictions, accept: str = "application/json"):
    return json.dumps({"predictions": predictions}), "application/json"
