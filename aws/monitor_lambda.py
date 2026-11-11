"""Weekly monitoring job (AWS Lambda handler, triggered by an EventBridge schedule).

1. Loads the training reference sample and the last week of captured, scored traffic from S3.
2. Joins confirmed fraud labels (chargebacks) that have arrived since.
3. Computes feature PSI, score PSI, recall, and missed fraud dollars by merchant category,
   using the same code as the local pipeline (fraud/monitoring.py).
4. Publishes them as CloudWatch metrics; CloudWatch alarms notify the team through SNS.

The local run showed why step 2 matters: when a new fraud pattern appeared, input PSI stayed
below 0.1 and only the outcome metrics moved.

Status: written for AWS Lambda (package with src/ and a pandas/scikit-learn layer) and not yet executed.
"""

from __future__ import annotations

import io
import json
import os

import boto3
import numpy as np
import pandas as pd

from fraud.monitoring import weekly_report

S3 = boto3.client("s3")
CW = boto3.client("cloudwatch")
BUCKET = os.environ.get("BUCKET", "")
NAMESPACE = "CardFraud/Model"


def _read_csv(key: str) -> pd.DataFrame:
    obj = S3.get_object(Bucket=BUCKET, Key=key)
    return pd.read_csv(io.BytesIO(obj["Body"].read()))


def handler(event, context):
    ref = _read_csv("fraud/monitoring/reference_sample.csv")          # features + score from training
    cur = _read_csv(event.get("current_key", "fraud/monitoring/last_week_scored.csv"))
    labels = _read_csv("fraud/labels/chargebacks.csv")                 # transaction_id, is_fraud
    cur = cur.drop(columns=["is_fraud"], errors="ignore").merge(labels, on="transaction_id", how="left")
    cur["is_fraud"] = cur["is_fraud"].fillna(0).astype(int)
    cur["week"] = 1
    threshold = float(json.loads(S3.get_object(Bucket=BUCKET, Key="fraud/monitoring/threshold.json")["Body"].read())["threshold"])

    rep = weekly_report(ref, ref["fraud_score"].to_numpy(), cur.reset_index(drop=True),
                        cur["fraud_score"].to_numpy(), threshold).iloc[0]
    metrics = [
        ("MaxFeaturePSI", rep.max_feature_psi, "None"),
        ("ScorePSI", rep.score_psi, "None"),
        ("FraudRecall", rep.fraud_recall if not np.isnan(rep.fraud_recall) else 0.0, "None"),
        ("MissedFraudDollars", rep.missed_fraud_dollars, "None"),
    ]
    CW.put_metric_data(Namespace=NAMESPACE, MetricData=[
        {"MetricName": n, "Value": float(v), "Unit": u} for n, v, u in metrics])
    return {"statusCode": 200, "body": rep.to_json()}
