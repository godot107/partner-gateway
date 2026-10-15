"""Simulated partners and the drift-detection experiment.

Partners send rows drawn from the held-out test set, so the true price of every house they ask about is known.
Two kinds of change, injected at a known request:

  units    Partner B's system starts sending median income in $100,000s instead of $10,000s (income x 0.1), or a
           milder scaling. Every request still passes the schema and returns 200; the predictions are just wrong.
  market   Partner B expands to one region (the Bay Area). The data is valid and the model may still be fine there:
           drift is a reason to look, not proof of harm.

`experiment()` measures, over many simulated runs: false alarms before the change, how many requests after the
change until the monitor flags it, and the harm meanwhile (prediction error against the true prices).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import drift
from .schema import to_csv
from .config import FEATURES, FIGURES, MODELS, RESULTS, SEED

BAY_AREA = {"Latitude": (37.0, 38.5), "Longitude": (-123.0, -121.5)}


def pool() -> pd.DataFrame:
    return pd.read_csv(MODELS / "partner_pool.csv")


def shifted(df: pd.DataFrame, kind: str, severity: float) -> pd.DataFrame:
    """The rows a partner sends after the change (`price` stays the truth for the house)."""
    if kind == "units":
        out = df.copy()
        out["MedInc"] = out["MedInc"] * severity
        return out
    if kind == "market":
        lat, lon = BAY_AREA["Latitude"], BAY_AREA["Longitude"]
        local = df[df.Latitude.between(*lat) & df.Longitude.between(*lon)]
        return _mix(df, local, severity)                      # severity = share of traffic from the new market
    raise ValueError(kind)


def _mix(base, local, share, n=5000, seed=SEED):
    rng = np.random.default_rng(seed)
    k = int(n * share)
    return pd.concat([local.iloc[rng.integers(0, len(local), k)], base.iloc[rng.integers(0, len(base), n - k)]])


def stream(df: pd.DataFrame, rng, n_requests: int, rows_per_request: int = 5):
    for _ in range(n_requests):
        yield df.iloc[rng.integers(0, len(df), rows_per_request)]


def one_run(model, ref, base, after, seed, before=100, after_n=200, rows_per_request=5, window=300, every=5):
    rng = np.random.default_rng(seed)
    mon = drift.Monitor(ref, window)
    false_alarm, detected_at, errs_before, errs_after = False, None, [], []
    for t, batch in enumerate(list(stream(base, rng, before, rows_per_request)) +
                              list(stream(after, rng, after_n, rows_per_request))):
        p = np.array([float(v) for v in model.predict_csv(to_csv(batch[FEATURES].to_dict("records"))).split()])
        (errs_before if t < before else errs_after).extend(np.abs(p - batch.price.to_numpy()))
        mon.add("b", batch[FEATURES].to_dict("records"), p.tolist())
        if t % every == 0:
            status = mon.check("b")["status"]
            if status == "drift":
                if t < before:
                    false_alarm = True
                elif detected_at is None:
                    detected_at = t - before
    return {"false alarm before change": false_alarm, "requests to detect": detected_at,
            "MAE before ($100k)": float(np.mean(errs_before)), "MAE after ($100k)": float(np.mean(errs_after))}


def experiment(sims=40, workers=1):
    from .model import Local
    model = Local()
    ref = json.loads((MODELS / "reference.json").read_text())
    base = pool()
    rows = []
    cases = [("none", "units", 1.0), ("units", "units", 0.9), ("units", "units", 0.75), ("units", "units", 0.5),
             ("units", "units", 0.1), ("market", "market", 0.3), ("market", "market", 1.0)]
    for label, kind, sev in cases:
        after = shifted(base, kind, sev)
        for s in range(sims):
            r = one_run(model, ref, base, after, SEED + s)
            rows.append({"case": label, "severity": sev, **r})
    d = pd.DataFrame(rows)
    g = d.groupby(["case", "severity"], sort=False)
    out = g.agg(**{"false alarms before change": ("false alarm before change", "mean"),
                   "detected within 200 requests": ("requests to detect", lambda x: x.notna().mean()),
                   "median requests to detect": ("requests to detect", "median"),
                   "MAE before ($100k)": ("MAE before ($100k)", "mean"),
                   "MAE after ($100k)": ("MAE after ($100k)", "mean")}).reset_index()
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "drift_experiment.json").write_text(json.dumps(out.to_dict(orient="records"), indent=1))
    with pd.option_context("display.width", 200, "display.precision", 3):
        print(out)
    plot(out)
    return out


def plot(out):
    import matplotlib.pyplot as plt
    labels = [f"income x{s:g}" if c == "units" else (f"new market {s:.0%}" if c == "market" else "no change")
              for c, s in zip(out.case, out.severity)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    y = np.arange(len(out))[::-1]
    axes[0].barh(y, out["median requests to detect"].fillna(0), color="#2a6fbb")
    for yi, (v, share) in zip(y, zip(out["median requests to detect"], out["detected within 200 requests"])):
        axes[0].text((0 if np.isnan(v) else v) + 1, yi, "not detected" if np.isnan(v) else f"{share:.0%} caught",
                     va="center", fontsize=8)
    axes[0].set_yticks(y, labels)
    axes[0].set_xlabel("median requests after the change until flagged (5 rows each)")
    axes[0].set_title("How fast is a change caught?", fontsize=10)
    axes[1].barh(y + 0.18, out["MAE before ($100k)"] * 100, 0.36, label="before the change", color="0.6")
    axes[1].barh(y - 0.18, out["MAE after ($100k)"] * 100, 0.36, label="after", color="#b8422e")
    axes[1].set_yticks(y, labels)
    axes[1].set_xlabel("mean absolute error of predicted price ($1,000s)")
    axes[1].set_title("What it costs meanwhile", fontsize=10)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "drift_experiment.png", dpi=130)
    plt.close(fig)
