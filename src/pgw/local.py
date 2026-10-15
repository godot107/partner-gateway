"""An offline stand-in for Cognito + API Gateway, so the whole demo runs without AWS.

It reproduces the checks in the order API Gateway applies them for this API:
  1. Cognito authorizer: is there a valid, unexpired token?            no -> 401
  2. scope on the method: does the token carry inference/predict?       no -> 403 (AWS may answer 401; the AWS
                                                                        demo prints whatever it gets)
  3. API key (system route only): present, known and enabled?           no -> 403
  4. usage plan: token-bucket rate limit and daily quota per key        over -> 429
  5. request schema                                                     invalid -> 400
then hands over to `service` (the same code the Lambda runs) with the local XGBoost model.

Tokens here are opaque random strings looked up in a table; real Cognito tokens are signed JWTs that API Gateway
verifies. The behaviour visible to a caller is the same.
"""
from __future__ import annotations

import json
import secrets
import time
import uuid

from . import drift, schema, service
from .config import MODELS, PARTNERS, SCOPES


class Bucket:
    """API Gateway's throttling model: `rate` requests/second refill, up to `burst`; plus a daily quota."""

    def __init__(self, rate, burst, quota, clock):
        self.rate, self.burst, self.quota, self.clock = rate, burst, quota, clock
        self.tokens, self.last, self.used = float(burst), clock(), 0

    def take(self) -> bool:
        now = self.clock()
        self.tokens = min(self.burst, self.tokens + (now - self.last) * self.rate)
        self.last = now
        if self.tokens < 1 or self.used >= self.quota:
            return False
        self.tokens -= 1
        self.used += 1
        return True


class LocalGateway:
    def __init__(self, partners_cfg: dict, model=None, clock=time.monotonic, token_ttl=3600):
        from .model import Local
        self.model = model or Local()
        self.clock, self.ttl = clock, token_ttl
        self.clients = {f"client-{p}": {"partner": p, "secret": secrets.token_hex(8),
                                        "scopes": {SCOPES["predict"], SCOPES["usage"]}} for p in PARTNERS}
        self.keys = {f"key-{p}": {"partner": p, "enabled": True} for p in PARTNERS}
        self.buckets = {k: Bucket(partners_cfg["partners"][v["partner"]]["rate_per_s"],
                                  partners_cfg["partners"][v["partner"]]["burst"],
                                  partners_cfg["partners"][v["partner"]]["quota_per_day"], clock)
                        for k, v in self.keys.items()}
        self.tokens: dict[str, dict] = {}
        self.logs: list[dict] = []
        self.monitor = drift.Monitor(json.loads((MODELS / "reference.json").read_text()))

    # --- Cognito ---------------------------------------------------------------------------------------------
    def credentials(self, partner):
        cid = f"client-{partner}"
        return cid, self.clients[cid]["secret"], f"key-{partner}"

    def token(self, client_id, secret, scope=None) -> tuple[int, dict]:
        """OAuth client-credentials grant at Cognito's /oauth2/token."""
        c = self.clients.get(client_id)
        if not c or c["secret"] != secret:
            return 400, {"error": "invalid_client"}
        want = set(scope.split()) if scope else c["scopes"]
        if not want <= c["scopes"]:
            return 400, {"error": "invalid_scope"}
        tok = secrets.token_urlsafe(24)
        self.tokens[tok] = {"client_id": client_id, "scope": " ".join(sorted(want)), "exp": self.clock() + self.ttl}
        return 200, {"access_token": tok, "token_type": "Bearer", "expires_in": self.ttl}

    # --- API Gateway -----------------------------------------------------------------------------------------
    def set_key_enabled(self, partner, enabled: bool):
        self.keys[f"key-{partner}"]["enabled"] = enabled

    def predict(self, body, token=None, api_key=None) -> tuple[int, dict]:
        claims = self.tokens.get(token or "")
        if not claims or claims["exp"] < self.clock():
            return 401, {"message": "Unauthorized"}
        if SCOPES["predict"] not in claims["scope"].split():
            return 403, {"message": "Forbidden: token lacks scope inference/predict"}
        key = self.keys.get(api_key or "")
        if not key or not key["enabled"]:
            return 403, {"message": "Forbidden"}
        if not self.buckets[api_key].take():
            return 429, {"message": "Too Many Requests"}
        errs = schema.errors(body)
        if errs:
            return 400, {"message": "Invalid request body", "errors": errs[:3]}
        who = service.caller(claims, {c: v["partner"] for c, v in self.clients.items()})
        r = service.predict(body, who, self.model.predict_csv, self._log, str(uuid.uuid4()))
        return r["statusCode"], json.loads(r["body"])

    def _log(self, record):
        self.logs.append(record)
        self.monitor.add(record["partner"], record["rows"], record["predictions"])

    # --- the drift job ---------------------------------------------------------------------------------------
    def drift_check(self) -> dict:
        return {p: self.monitor.check(p) for p in PARTNERS}
