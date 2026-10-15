"""What happens after API Gateway lets a request through. Pure Python, shared by the Lambda and the local gateway.

API Gateway has already checked the token (signature, expiry, scope), the partner's API key and usage plan, and the
request schema. This layer answers the questions the gateway can't:
  * who is this, as a partner? (a system's token carries its app client ID; a person's carries their Cognito groups)
  * may they see this? (a partner's people see only their own partner's usage and drift; admins see all)
and then calls the model and logs the request under the partner, which is what drift monitoring reads.
"""
from __future__ import annotations

import json
import time

from .config import FEATURES, PARTNERS
from .schema import quick_check, to_csv


def caller(claims: dict, clients: dict[str, str]) -> dict:
    """From the authorizer's token claims to {partner, kind, admin}. `clients` maps app client ID -> partner."""
    groups = claims.get("cognito:groups", "")
    groups = groups if isinstance(groups, list) else [g for g in str(groups).strip("[]").replace(",", " ").split() if g]
    if "username" in claims and groups:                                  # a person, signed in on the web page
        partner = next((g for g in groups if g in PARTNERS), None)
        return {"partner": partner, "kind": "user", "admin": "admin" in groups, "user": claims.get("username")}
    cid = claims.get("client_id")
    return {"partner": clients.get(cid), "kind": "system", "admin": False, "user": cid}


def response(status: int, body: dict) -> dict:
    return {"statusCode": status, "headers": {"Content-Type": "application/json"}, "body": json.dumps(body)}


def predict(body, who: dict, invoke, log, request_id: str, now=time.time) -> dict:
    """invoke(csv) -> csv of predictions (SageMaker or the local model); log(record) stores the request."""
    if who["partner"] is None:
        return response(403, {"message": "this caller is not mapped to a partner"})
    problem = quick_check(body)
    if problem:
        return response(400, {"message": problem})
    t0 = now()
    preds = [float(v) for v in invoke(to_csv(body["rows"])).split()]
    record = {"ts": t0, "request_id": request_id, "partner": who["partner"], "caller": who["kind"],
              "rows": [{f: r[f] for f in FEATURES} for r in body["rows"]], "predictions": preds}
    log(record)
    return response(200, {"partner": who["partner"], "request_id": request_id,
                          "predictions": [{"price_usd": round(p * 100_000)} for p in preds]})


def usage(who: dict, summary: dict) -> dict:
    """summary: {partner: {...requests, rows, drift verdict...}}. People see their own partner; admins see all."""
    if who["admin"]:
        return response(200, summary)
    if who["partner"] is None:
        return response(403, {"message": "not a member of a partner group"})
    return response(200, {who["partner"]: summary.get(who["partner"], {})})
