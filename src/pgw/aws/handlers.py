"""Lambda entry points (python3.12, standard library + the runtime's boto3). Thin: the logic is in pgw.service and
pgw.drift, which the local demo exercises too.

  predict   POST {stage}/predict (partner systems: token + API key) and POST {stage}/me/predict (people)
  usage     GET  {stage}/me/usage: request counts and the latest drift verdict, own partner only (admins: all)
  web       GET  /: the sign-in page for people (Cognito managed login, authorization code + PKCE)
  drift     scheduled, or invoked by the demo: score each partner's recent traffic, alert on a new drift

Environment: ENDPOINT, BUCKET, CLIENTS (JSON: app client ID -> partner), TOPIC_ARN, and for the web page
AUTH_DOMAIN, WEB_CLIENT_ID, API_URL.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import boto3

from pgw import drift, service
from pgw.config import PARTNERS

_s3 = boto3.client("s3")
_sm = boto3.client("sagemaker-runtime")
WINDOW = 300
_REF = None


def _ref():
    """The drift reference, packaged next to the code (deploy.sh copies models/reference.json in)."""
    global _REF
    if _REF is None:
        here = Path(__file__).resolve().parents[1]
        path = here / "reference.json"
        if not path.exists():                                   # running from the repo (tests)
            path = here.parents[1] / "models" / "reference.json"
        _REF = json.loads(path.read_text())
    return _REF


def _env(k):
    return os.environ[k]


def _claims(event):
    return (event.get("requestContext") or {}).get("authorizer", {}).get("claims", {}) or {}


def _metric(name, value, unit, partner):
    """CloudWatch Embedded Metric Format: a log line that becomes a metric, no API call."""
    print(json.dumps({"_aws": {"Timestamp": int(time.time() * 1000), "CloudWatchMetrics": [
        {"Namespace": "partner-gateway", "Dimensions": [["Partner"]], "Metrics": [{"Name": name, "Unit": unit}]}]},
        "Partner": partner, name: value}))


def _invoke(csv):
    r = _sm.invoke_endpoint(EndpointName=_env("ENDPOINT"), ContentType="text/csv", Accept="text/csv", Body=csv)
    return r["Body"].read().decode()


def _log(record):
    day = datetime.fromtimestamp(record["ts"], timezone.utc).strftime("%Y-%m-%d")
    key = f"logs/{record['partner']}/{day}/{int(record['ts'] * 1000):013d}-{record['request_id']}.json"
    _s3.put_object(Bucket=_env("BUCKET"), Key=key, Body=json.dumps(record).encode(), ContentType="application/json")
    _metric("Rows", len(record["rows"]), "Count", record["partner"])


def predict(event, context):
    who = service.caller(_claims(event), json.loads(_env("CLIENTS")))
    try:
        body = json.loads(event.get("body") or "null")
    except json.JSONDecodeError:
        return service.response(400, {"message": "body is not JSON"})
    t0 = time.time()
    r = service.predict(body, who, _invoke, _log, context.aws_request_id)
    if who["partner"]:
        _metric("LatencyMs", (time.time() - t0) * 1000, "Milliseconds", who["partner"])
    return r


def _summary():
    try:
        return json.loads(_s3.get_object(Bucket=_env("BUCKET"), Key="drift/latest.json")["Body"].read())
    except _s3.exceptions.NoSuchKey:
        return {}


def usage(event, context):
    return service.usage(service.caller(_claims(event), json.loads(_env("CLIENTS"))), _summary())


def web(event, context):
    here = Path(__file__).resolve().parents[1]
    page = here / "index.html"
    html = (page if page.exists() else here.parents[1] / "web" / "index.html").read_text()
    for k in ("AUTH_DOMAIN", "WEB_CLIENT_ID", "API_URL"):
        html = html.replace("{{" + k + "}}", _env(k))
    return {"statusCode": 200, "headers": {"Content-Type": "text/html; charset=utf-8"}, "body": html}


def _recent(partner, now):
    """The newest log records for a partner (today and yesterday, UTC), up to WINDOW rows."""
    keys = []
    for day in (now, now - timedelta(days=1)):
        prefix = f"logs/{partner}/{day:%Y-%m-%d}/"
        for page in _s3.get_paginator("list_objects_v2").paginate(Bucket=_env("BUCKET"), Prefix=prefix):
            keys += [o["Key"] for o in page.get("Contents", [])]
    rows, preds, today = [], [], sum(1 for k in keys if f"/{now:%Y-%m-%d}/" in k)
    for k in sorted(keys, reverse=True):
        rec = json.loads(_s3.get_object(Bucket=_env("BUCKET"), Key=k)["Body"].read())
        rows += rec["rows"]
        preds += rec["predictions"]
        if len(rows) >= WINDOW:
            break
    return rows[:WINDOW], preds[:WINDOW], today


def drift_job(event, context):
    now = datetime.now(timezone.utc)
    before = _summary()
    out = {}
    for p in PARTNERS:
        rows, preds, today = _recent(p, now)
        v = drift.verdict(drift.score(_ref(), rows, preds) if rows else {}, len(rows))
        out[p] = {**v, "requests_today": today, "checked_at": now.isoformat(timespec="seconds")}
        _metric("WorstPSI", v["worst_psi"], "None", p)
        if v["status"] == "drift" and before.get(p, {}).get("status") != "drift":
            boto3.client("sns").publish(
                TopicArn=_env("TOPIC_ARN"), Subject=f"Drift detected: {p}",
                Message=(f"{p}'s recent traffic no longer looks like the training data.\n"
                         f"Worst input: {v['worst_feature']} (PSI {v['worst_psi']:.2f}; above "
                         f"{drift.THRESHOLD} means shifted), over the last {v['rows']} rows.\n"
                         "Requests still succeed; check what changed on the partner's side before trusting the "
                         "predictions."),
                MessageAttributes={"partner": {"DataType": "String", "StringValue": p}})
    _s3.put_object(Bucket=_env("BUCKET"), Key="drift/latest.json", Body=json.dumps(out).encode(),
                   ContentType="application/json")
    return out
