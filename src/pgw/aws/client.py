"""The demo's view of the deployed stack: the same calls as pgw.local.LocalGateway, over HTTPS.

Secrets are never stored: client secrets and API key values are read from AWS at run time with the operator's
own credentials (`aws login`), from the stack's outputs.
"""
from __future__ import annotations

import base64
import json
import urllib.error
import urllib.parse
import urllib.request

import boto3

STACK = "partner-gateway-app"


def _http(url, data=None, headers=None, method=None):
    req = urllib.request.Request(url, data=data, headers=headers or {}, method=method)
    try:
        with urllib.request.urlopen(req, timeout=35) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read() or b"{}")
        except json.JSONDecodeError:
            return e.code, {}


class AwsGateway:
    def __init__(self, outputs: dict, region: str):
        self.o, self.region = outputs, region
        self.cog = boto3.client("cognito-idp", region_name=region)
        self.apigw = boto3.client("apigateway", region_name=region)
        self.lam = boto3.client("lambda", region_name=region)

    @classmethod
    def from_stack(cls, stack=STACK):
        cf = boto3.client("cloudformation")
        s = cf.describe_stacks(StackName=stack)["Stacks"][0]
        return cls({o["OutputKey"]: o["OutputValue"] for o in s["Outputs"]}, cf.meta.region_name)

    def credentials(self, partner):
        tag = {"partner-a": "A", "partner-b": "B"}[partner]
        cid = self.o[f"ClientId{tag}"]
        secret = self.cog.describe_user_pool_client(UserPoolId=self.o["UserPoolId"],
                                                    ClientId=cid)["UserPoolClient"]["ClientSecret"]
        key = self.apigw.get_api_key(apiKey=self.o[f"ApiKeyId{tag}"], includeValue=True)["value"]
        self._key_ids = {**getattr(self, "_key_ids", {}), partner: self.o[f"ApiKeyId{tag}"]}
        return cid, secret, key

    def token(self, client_id, secret, scope=None):
        """Client-credentials grant at Cognito's token endpoint."""
        form = {"grant_type": "client_credentials", **({"scope": scope} if scope else {})}
        basic = base64.b64encode(f"{client_id}:{secret}".encode()).decode()
        return _http(self.o["TokenUrl"], urllib.parse.urlencode(form).encode(),
                     {"Authorization": f"Basic {basic}", "Content-Type": "application/x-www-form-urlencoded"})

    def predict(self, body, token=None, api_key=None):
        headers = {"Content-Type": "application/json"}
        if token:
            headers["Authorization"] = token
        if api_key:
            headers["x-api-key"] = api_key
        return _http(self.o["ApiUrl"].rstrip("/") + "/predict", json.dumps(body).encode(), headers, "POST")

    def set_key_enabled(self, partner, enabled: bool):
        self.apigw.update_api_key(apiKey=self._key_ids[partner],
                                  patchOperations=[{"op": "replace", "path": "/enabled", "value": str(enabled).lower()}])
        import time
        time.sleep(15)                     # key changes take a few seconds to reach API Gateway's edge

    def drift_check(self):
        r = self.lam.invoke(FunctionName=self.o["DriftFunction"])
        return json.loads(r["Payload"].read())
