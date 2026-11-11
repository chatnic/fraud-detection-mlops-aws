"""Local end-to-end run: features -> time-based split -> train -> evaluate at analyst capacity
-> monitor the drift period -> retrain on fresh labels.

    PYTHONPATH=src python -m fraud.pipeline --data data/raw/transactions.csv --out reports
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from fraud.evaluate import ALERT_RATE, alert_metrics, recall_by_type, threshold_for_alert_rate  # noqa: E402
from fraud.features import FEATURES, build_features  # noqa: E402
from fraud.model import rule_score, score, train  # noqa: E402
from fraud.monitoring import weekly_report  # noqa: E402

TRAIN_END, VAL_END, RETRAIN_END = 56, 69, 83   # day boundaries


def run(data: Path, out: Path) -> dict:
    raw = pd.read_csv(data, parse_dates=["timestamp"])
    df = build_features(raw)
    day = (df["timestamp"] - df["timestamp"].min().normalize()).dt.days
    df["day"] = day
    df["week"] = (day - VAL_END) // 7 + 1          # test weeks 1, 2, 3
    tr, va, te = df[day < TRAIN_END], df[(day >= TRAIN_END) & (day < VAL_END)], df[day >= VAL_END]
    print(f"train {len(tr):,} (days 0-{TRAIN_END - 1})  validation {len(va):,}  test {len(te):,} (days {VAL_END}+)")

    model = train(tr)
    thr = threshold_for_alert_rate(score(model, va))
    s_te = score(model, te)
    m_model = alert_metrics(te, s_te, thr)

    # Rule baseline: evaluated at its own alert volume, then the model at that same volume.
    r_te = rule_score(te)
    m_rules = alert_metrics(te, r_te, 0.5)
    thr_same = float(np.quantile(s_te, 1 - m_rules["alert_rate"]))
    m_model_same = alert_metrics(te, s_te, thr_same)
    print(f"rules: alert rate {m_rules['alert_rate']:.2%}, recall {m_rules['recall']:.1%}, precision {m_rules['precision']:.1%}")
    print(f"model @ same volume: recall {m_model_same['recall']:.1%}, precision {m_model_same['precision']:.1%}")
    print(f"model @ {ALERT_RATE:.1%} capacity: recall {m_model['recall']:.1%}, $ recall {m_model['dollar_recall']:.1%}, "
          f"precision {m_model['precision']:.1%}")

    pre_drift = te[te.day < RETRAIN_END - 0]
    types_test = recall_by_type(te, s_te, thr)

    # Monitoring over the test weeks (reference = training data).
    s_tr = score(model, tr)
    te_r = te.reset_index(drop=True)
    weekly = weekly_report(tr, s_tr, te_r, score(model, te_r), thr)

    # Retrain once a week of crypto chargebacks has arrived, compare on the remaining days.
    later = df[day >= RETRAIN_END]
    model2 = train(df[day < RETRAIN_END - 7].pipe(lambda d: pd.concat([d, df[(day >= RETRAIN_END - 7) & (day < RETRAIN_END)]])))
    va2 = df[(day >= RETRAIN_END - 7) & (day < RETRAIN_END)]
    thr2 = threshold_for_alert_rate(score(model2, va2))
    before = recall_by_type(later, score(model, later), thr)
    after = recall_by_type(later, score(model2, later), thr2)
    m_before = alert_metrics(later, score(model, later), thr)
    m_after = alert_metrics(later, score(model2, later), thr2)

    metrics = {
        "split_days": {"train": [0, TRAIN_END - 1], "validation": [TRAIN_END, VAL_END - 1], "test": [VAL_END, int(day.max())]},
        "fraud_rate": float(df.is_fraud.mean()),
        "alert_capacity": ALERT_RATE,
        "rules": m_rules,
        "model_same_alert_volume_as_rules": m_model_same,
        "model_at_capacity": m_model,
        "recall_by_type_test": types_test.round(3).to_dict(),
        "retraining": {
            "evaluated_on_days": [RETRAIN_END, int(day.max())],
            "recall_before": m_before["recall"], "recall_after": m_after["recall"],
            "dollar_recall_before": m_before["dollar_recall"], "dollar_recall_after": m_after["dollar_recall"],
            "crypto_recall_before": float(before.get("crypto_cashout", np.nan)),
            "crypto_recall_after": float(after.get("crypto_cashout", np.nan)),
            "by_type_before": before.round(3).to_dict(), "by_type_after": after.round(3).to_dict(),
        },
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(exist_ok=True)
    weekly.round(4).to_csv(out / "tables" / "weekly_monitoring.csv", index=False)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    Path("models").mkdir(exist_ok=True)
    joblib.dump({"model": model2, "threshold": thr2, "features": FEATURES}, "models/fraud_model.joblib")
    figures(m_rules, m_model_same, weekly, before, after, out / "figures")
    print(weekly.round(3).to_string())
    print(json.dumps(metrics["retraining"], indent=2, default=float))
    return metrics


def figures(m_rules, m_same, weekly, before, after, fig_dir: Path) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 10})

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    for ax, key, title in [(axes[0], "recall", "Fraud transactions caught"),
                           (axes[1], "precision", "Alerts that are real fraud")]:
        vals = [m_rules[key], m_same[key]]
        ax.bar(["Legacy rules", "Model"], vals, color=["#9aa5b1", "#1f4e8c"])
        for i, v in enumerate(vals):
            ax.text(i, v + 0.01, f"{v:.0%}", ha="center")
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.set_title(title)
    fig.suptitle(f"Same alert volume ({m_rules['alert_rate']:.2%} of transactions), test period")
    fig.tight_layout(); fig.savefig(fig_dir / "rules_vs_model.png", dpi=150); plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(weekly.week, weekly.fraud_recall, marker="o", color="#1f4e8c", label="Fraud recall (from chargebacks)")
    ax1.set_ylabel("Recall"); ax1.set_ylim(0, 1)
    ax1.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax2 = ax1.twinx()
    ax2.plot(weekly.week, weekly.max_feature_psi, marker="s", color="#e08a2c", label="Max feature PSI")
    ax2.axhline(0.25, color="#e08a2c", ls=":", lw=1)
    ax2.set_ylabel("PSI"); ax2.set_ylim(0, max(0.5, weekly.max_feature_psi.max() * 1.2))
    ax1.set_xticks(weekly.week, [f"Test week {int(w)}" for w in weekly.week])
    ax1.set_title("Monitoring the drift period: inputs look stable, recall drops")
    lines = ax1.get_lines() + ax2.get_lines()[:1]
    ax1.legend(lines, [l.get_label() for l in lines], frameon=False, loc="center right")
    fig.tight_layout(); fig.savefig(fig_dir / "monitoring.png", dpi=150); plt.close(fig)

    types = sorted(set(before.index) | set(after.index))
    x = np.arange(len(types))
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.bar(x - 0.18, [before.get(t, 0) for t in types], 0.36, label="Original model", color="#9aa5b1")
    ax.bar(x + 0.18, [after.get(t, 0) for t in types], 0.36, label="Retrained with fresh labels", color="#1f4e8c")
    ax.set_xticks(x, [t.replace("_", " ") for t in types])
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_ylim(0, 1.3)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylabel("Recall at analyst capacity")
    ax.set_title("Recall by fraud type, final week (after retraining)")
    ax.legend(frameon=False, loc="upper center", ncol=2)
    fig.tight_layout(); fig.savefig(fig_dir / "retraining.png", dpi=150); plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("data/raw/transactions.csv"))
    p.add_argument("--out", type=Path, default=Path("reports"))
    a = p.parse_args()
    run(a.data, a.out)


if __name__ == "__main__":
    main()
