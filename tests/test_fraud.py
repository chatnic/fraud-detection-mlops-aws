import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "data"))

from generate_data import generate  # noqa: E402
from fraud.evaluate import alert_metrics, threshold_for_alert_rate  # noqa: E402
from fraud.features import FEATURES, build_features  # noqa: E402
from fraud.model import score, train  # noqa: E402
from fraud.monitoring import psi  # noqa: E402
import inference_entry  # noqa: E402


@pytest.fixture(scope="module")
def tx():
    return generate(cardholders=300, days=30, seed=3)


@pytest.fixture(scope="module")
def feats(tx):
    return build_features(tx)


def test_features_are_point_in_time(tx):
    """Features for early transactions must not change when later transactions are added."""
    cutoff = tx["timestamp"].min() + pd.Timedelta(days=15)
    early_only = build_features(tx[tx.timestamp < cutoff]).set_index("transaction_id")[FEATURES]
    full = build_features(tx).set_index("transaction_id").loc[early_only.index, FEATURES]
    pd.testing.assert_frame_equal(early_only.sort_index(), full.sort_index(), check_dtype=False)


def test_velocity_counts_prior_only(feats):
    first = feats.sort_values(["timestamp", "transaction_id"]).groupby("card_id").head(1)
    assert (first["txn_count_1h"] == 0).all() and (first["card_txn_count"] == 0).all()


def test_psi_behaviour():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 20_000)
    assert psi(a, rng.normal(0, 1, 20_000)) < 0.02
    assert psi(a, rng.normal(1, 1, 20_000)) > 0.25
    assert psi(np.array(list("aab") * 100), np.array(list("abb") * 100), categorical=True) > 0.1


def test_alert_metrics_and_threshold(feats):
    s = np.linspace(0, 1, len(feats))
    thr = threshold_for_alert_rate(s, 0.1)
    m = alert_metrics(feats, s, thr)
    assert m["alert_rate"] == pytest.approx(0.1, abs=0.01)


def test_inference_handlers_roundtrip(feats, tmp_path):
    import joblib
    model = train(feats)
    joblib.dump({"model": model, "threshold": 0.5, "features": FEATURES}, tmp_path / "fraud_model.joblib")
    bundle = inference_entry.model_fn(str(tmp_path))
    body = json.dumps({"instances": feats[FEATURES].head(3).to_dict(orient="records")})
    preds = inference_entry.predict_fn(inference_entry.input_fn(body, "application/json"), bundle)
    out, ctype = inference_entry.output_fn(preds)
    parsed = json.loads(out)["predictions"]
    assert len(parsed) == 3 and set(parsed[0]) == {"fraud_score", "alert"}
    assert np.allclose([p["fraud_score"] for p in parsed], score(model, feats.head(3)))
