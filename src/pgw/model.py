"""Train the home-price model and package it for SageMaker's built-in XGBoost container.

The model is saved in XGBoost's own format (file name `xgboost-model`, which the container loads); no pickle.
A reference profile of the training features (`reference.json`: quantile bin edges and shares per feature, plus the
prediction distribution) is written next to it: it is what drift monitoring compares partners' traffic with.
"""
from __future__ import annotations

import json
import tarfile

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

from . import drift
from .config import DATA, FEATURES, MODELS, RESULTS, SEED

PARAMS = {"objective": "reg:squarederror", "eta": 0.05, "max_depth": 6, "subsample": 0.8,
          "colsample_bytree": 0.8, "seed": SEED}
ROUNDS = 600


def load() -> pd.DataFrame:
    d = fetch_california_housing(data_home=str(DATA / "sk"), as_frame=True)
    return d.frame.rename(columns={"MedHouseVal": "price"})


def split(df):
    """Train / validation / test, 60/20/20. The test rows are what simulated partners send."""
    train, rest = train_test_split(df, test_size=0.4, random_state=SEED)
    val, test = train_test_split(rest, test_size=0.5, random_state=SEED)
    return train, val, test


def train():
    df = load()
    tr, va, te = split(df)
    # Plain arrays, no column names: the container builds an unnamed DMatrix from CSV, and a booster trained
    # with names can reject unnamed input. Column order is FEATURES everywhere.
    dtr = xgb.DMatrix(tr[FEATURES].to_numpy(), tr.price.to_numpy())
    dva = xgb.DMatrix(va[FEATURES].to_numpy(), va.price.to_numpy())
    booster = xgb.train(PARAMS, dtr, ROUNDS, evals=[(dva, "val")], early_stopping_rounds=50, verbose_eval=False)
    best = booster.best_iteration + 1
    booster = booster[:best]                                     # keep the early-stopped trees only
    pred = booster.predict(xgb.DMatrix(te[FEATURES].to_numpy()))
    err = pred - te.price.to_numpy()
    metrics = {"rows": {"train": len(tr), "val": len(va), "test": len(te)}, "rounds": best,
               "test RMSE ($100k)": float(np.sqrt(np.mean(err ** 2))), "test MAE ($100k)": float(np.mean(np.abs(err))),
               "test R2": float(1 - np.mean(err ** 2) / np.var(te.price))}
    MODELS.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(MODELS / "xgboost-model"))          # XGBoost 1.7 binary format, no pickle
    ref = drift.reference({f: tr[f].tolist() for f in FEATURES} | {"prediction": booster.predict(dtr).tolist()})
    (MODELS / "reference.json").write_text(json.dumps(ref))
    te.to_csv(MODELS / "partner_pool.csv", index=False)         # rows the simulated partners draw from
    with tarfile.open(MODELS / "model.tar.gz", "w:gz") as t:
        t.add(MODELS / "xgboost-model", arcname="xgboost-model")
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "model.json").write_text(json.dumps(metrics, indent=1))
    return metrics


class Local:
    """The model as the endpoint would run it, for the local gateway and the experiments."""

    def __init__(self):
        self.b = xgb.Booster()
        self.b.load_model(str(MODELS / "xgboost-model"))

    def predict_csv(self, csv: str) -> str:
        X = np.array([[float(v) for v in line.split(",")] for line in csv.strip().splitlines()])
        p = self.b.predict(xgb.DMatrix(X))
        return "\n".join(f"{v:.6f}" for v in p)              # the container's text/csv response: one value per line
