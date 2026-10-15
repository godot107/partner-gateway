"""Build notebooks/walkthrough.ipynb, then execute it:

    PYTHONPATH=src .venv/bin/python notebooks/build_walkthrough.py
    PYTHONPATH=src .venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/walkthrough.ipynb

Runs the offline demo live; reads the detection experiment from reports/results (`python -m pgw experiment`).
"""
from pathlib import Path

import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

cells = [
    md("""# Opening a model to partners, safely

A company has a model that prices homes and wants to sell access to it: to a realtor (**Partner A**) and a lender
(**Partner B**). This notebook walks through what stands between the outside world and the model, and runs the
whole demo offline.

| Layer | Question it answers | AWS service |
|---|---|---|
| Identity | Who is calling? | Cognito: a token for each partner's system; sign-in for partner staff |
| Gateway | May they call this, this often, with this kind of request? | API Gateway: scopes, API keys, usage plans, request schema |
| Service | Which partner is it, and what may they see? | A small Lambda function |
| Model | What's the price? | SageMaker serverless endpoint |
| Monitoring | Has a partner's data changed? | Request logs + a drift check + email alerts |"""),
    code("""import json
import pandas as pd
from IPython.display import Image, display

from pgw import config, drift, schema
from pgw.config import MODELS, RESULTS, FIGURES

pd.set_option("display.precision", 3)
pd.set_option("display.width", 160)
print(json.dumps(json.loads((RESULTS / "model.json").read_text()), indent=1))"""),
    md("""The model: XGBoost on 1990 California census block groups. On held-out data it explains 85% of the variation
in prices, with a typical error of about $30,000. (Prices are in units of $100,000, the dataset's own unit.)

## 1. The contract

Every request must match a schema: eight numbers per house, each within a plausible range. API Gateway checks it
before anything runs, so a malformed request costs nothing. But look what a contract *can't* catch:"""),
    code("""house = {"MedInc": 4.2, "HouseAge": 25, "AveRooms": 5.4, "AveBedrms": 1.05, "Population": 1200,
         "AveOccup": 2.9, "Latitude": 34.05, "Longitude": -118.25}
print("valid house:                    ", schema.errors({"rows": [house]}) or "OK")
print("income as text:                 ", schema.errors({"rows": [{**house, "MedInc": "high"}]}))
print("income in $100k instead of $10k:", schema.errors({"rows": [{**house, "MedInc": 0.42}]}) or "OK (!)")"""),
    md("""Median income 0.42 is a perfectly plausible number (it means $4,200), so the request is valid. It's also wrong:
the partner meant $42,000. That's the gap the rest of the demo is about.

## 2. The demo, offline

`LocalGateway` reproduces Cognito and API Gateway's checks in the order AWS applies them (token, scope, API key,
usage plan, schema) and then runs the same service code the AWS Lambda runs, with the real model."""),
    code("""from pgw import demo
results = pd.DataFrame(demo.local(verbose=False))
results[["step", "title", "expected", "got", "ok", "note"]]"""),
    md("""- **Steps 1-2:** no token, or a token without the right permission (scope): stopped at the gateway.
- **Step 3:** Partner A gets a price.
- **Step 4:** a malformed request is rejected before the model runs.
- **Step 5:** Partner B sends 12 requests at once; its plan allows 1 per second with a burst of 2, so most get
  429 "Too Many Requests". Partner A, on its own plan, is unaffected.
- **Step 6:** switching off Partner B's API key cuts B off instantly without touching A. (Tokens alone can't be
  revoked that quickly: they stay valid until they expire.)
- **Step 7:** Partner B's system starts sending income in the wrong units. Every request **succeeds**. The prices
  are just badly wrong.
- **Step 8:** the drift check compares each partner's recent requests with the training data and flags B.

## 3. How the drift check works

For every input, the training data is split into ten equal-sized bins (deciles). A partner's recent requests are
binned the same way. The *population stability index* (PSI) adds up how far each bin's share has moved:
0 means identical, above 0.25 is usually read as "shifted"."""),
    code("""ref = json.loads((MODELS / "reference.json").read_text())
pool = pd.read_csv(MODELS / "partner_pool.csv").sample(300, random_state=0)
normal = drift.score(ref, pool[config.FEATURES].to_dict("records"))
wrong_units = drift.score(ref, pool.assign(MedInc=pool.MedInc * 0.1)[config.FEATURES].to_dict("records"))
pd.DataFrame({"normal traffic": normal, "income in wrong units": wrong_units}).round(3)"""),
    md("""Normal traffic stays near zero on every input; the unit mistake sends median income's PSI far past 0.25,
and only that input moves, which points straight at the field to check with the partner.

## 4. How fast is a change caught, and what does it cost meanwhile?

The simulated partners send real houses from data the model never trained on, so the true price of every request
is known. Each change starts after 100 normal requests; 40 runs per case."""),
    code("""pd.DataFrame(json.loads((RESULTS / "drift_experiment.json").read_text()))"""),
    code("""display(Image(str(FIGURES / "drift_experiment.png")))"""),
    md("""- **Harmful changes are caught quickly**: income at half or a tenth of its value, within 15 to 20 requests;
  a 25% error within 50. Meanwhile the typical price error rises from $30k to as much as $63k.
- **A harmless change is ignored**: income 10% low barely moves the error, and the monitor stays quiet.
- **Drift isn't the same as damage**: a partner moving into a new market (the Bay Area) trips the monitor, yet the
  model's error rises only 12%. The alert is a reason to look, not proof the model is wrong.
- **False alarms are real**: 7.5% of runs raised an alarm before anything changed. Tuning the warm-up and threshold
  trades false alarms against speed.

## 5. On AWS

The same eight steps run against the deployed stack with `python -m pgw demo-aws`, plus a sign-in page for partner
staff where each person sees only their own partner's usage and data health. Deployment needs an AWS login; the
results will be added here after the first run."""),
]

nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"}})
out = Path(__file__).with_name("walkthrough.ipynb")
nbf.write(nb, out)
print(out)
