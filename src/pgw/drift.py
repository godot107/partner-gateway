"""Drift monitoring per partner. Pure Python (no numpy), so the same code runs locally and in the Lambda.

Reference: for every feature (and the model's predictions), ten bins with edges at the training data's deciles.
Monitoring: for each partner, the last `window` requests' rows are binned the same way and compared with the
reference by the population stability index,

    PSI = sum over bins of (actual share - reference share) * ln(actual share / reference share).

Common reading (a rule of thumb from credit scoring): < 0.1 stable, 0.1-0.25 shifting, > 0.25 shifted. A partner
is flagged when any feature's PSI exceeds `THRESHOLD`. A window is only scored once it has `MIN_ROWS` rows,
because PSI on a few rows is mostly noise.
"""
from __future__ import annotations

import bisect
import math
from collections import deque

BINS = 10
THRESHOLD = 0.25
MIN_ROWS = 100
EPS = 1e-4


def _quantile(sorted_vals, q):
    i = q * (len(sorted_vals) - 1)
    lo, hi = math.floor(i), math.ceil(i)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (i - lo)


def reference(columns: dict[str, list[float]], bins: int = BINS) -> dict:
    """{name: {"edges": interior bin edges, "shares": share of training rows per bin}}."""
    out = {}
    for name, vals in columns.items():
        s = sorted(vals)
        edges = sorted(set(_quantile(s, k / bins) for k in range(1, bins)))
        out[name] = {"edges": edges, "shares": shares(vals, edges)}
    return out


def shares(vals, edges):
    counts = [0] * (len(edges) + 1)
    for v in vals:
        counts[bisect.bisect_right(edges, v)] += 1
    n = max(len(vals), 1)
    return [c / n for c in counts]


def psi(ref: dict, vals) -> float:
    a = shares(vals, ref["edges"])
    return sum((x - r) * math.log((x + EPS) / (r + EPS)) for x, r in zip(a, ref["shares"]))


def score(ref: dict, rows: list[dict], predictions: list[float] | None = None) -> dict:
    """PSI per feature (and for the predictions, if given) for a batch of request rows."""
    out = {f: psi(ref[f], [r[f] for r in rows]) for f in ref if f != "prediction"}
    if predictions is not None and "prediction" in ref:
        out["prediction"] = psi(ref["prediction"], predictions)
    return out


def verdict(scores: dict, n: int) -> dict:
    worst = max(scores, key=scores.get) if scores else None
    return {"rows": n, "status": "warming up" if n < MIN_ROWS else
            ("drift" if worst and scores[worst] > THRESHOLD else "ok"),
            "worst_feature": worst, "worst_psi": scores.get(worst, 0.0) if worst else 0.0, "psi": scores}


class Monitor:
    """Sliding window of the last `window` rows per partner."""

    def __init__(self, ref: dict, window: int = 300):
        self.ref, self.window = ref, window
        self.rows: dict[str, deque] = {}
        self.preds: dict[str, deque] = {}

    def add(self, partner: str, rows: list[dict], predictions: list[float]):
        self.rows.setdefault(partner, deque(maxlen=self.window)).extend(rows)
        self.preds.setdefault(partner, deque(maxlen=self.window)).extend(predictions)

    def check(self, partner: str) -> dict:
        rows = list(self.rows.get(partner, []))
        return verdict(score(self.ref, rows, list(self.preds.get(partner, []))) if rows else {}, len(rows))
