import io
import json
import os
import random

import pytest
import yaml

from pgw import drift, schema, service
from pgw.config import FEATURES, MODELS, ROOT

GOOD = {"MedInc": 4.2, "HouseAge": 25, "AveRooms": 5.4, "AveBedrms": 1.05, "Population": 1200, "AveOccup": 2.9,
        "Latitude": 34.05, "Longitude": -118.25}


@pytest.fixture(scope="session")
def trained():
    if not (MODELS / "reference.json").exists():
        from pgw import model
        model.train()
    return True


# ---- the contract ---------------------------------------------------------------------------------------------

def _cfn_yaml(path):
    class L(yaml.SafeLoader):
        pass
    for tag in ("!Ref", "!Sub", "!GetAtt", "!Join", "!If", "!Equals", "!Select", "!Split"):
        L.add_constructor(tag, lambda loader, node: None)
    return yaml.load(path.read_text(), Loader=L)


def test_gateway_schema_is_the_python_schema():
    t = _cfn_yaml(ROOT / "infra" / "app.yaml")
    assert t["Resources"]["PredictModel"]["Properties"]["Schema"] == schema.REQUEST


@pytest.mark.parametrize("body,ok", [
    ({"rows": [GOOD]}, True),
    ({"rows": [{**GOOD, "MedInc": "high"}]}, False),
    ({"rows": [{k: v for k, v in GOOD.items() if k != "Latitude"}]}, False),
    ({"rows": [{**GOOD, "MedInc": 99}]}, False),
    ({"rows": [{**GOOD, "MedInc": True}]}, False),
    ({"rows": [{**GOOD, "extra": 1}]}, False),
    ({"rows": []}, False),
    ({"rows": [GOOD] * 101}, False),
    ({"rows": [GOOD], "x": 1}, False),
])
def test_python_check_agrees_with_json_schema(body, ok):
    assert (schema.errors(body) == []) == ok
    assert (schema.quick_check(body) is None) == ok


def test_wrong_units_pass_the_contract():
    """The point of the demo: income in $100k units is still a valid request."""
    assert schema.quick_check({"rows": [{**GOOD, "MedInc": GOOD["MedInc"] * 0.1}]}) is None


def test_csv_is_in_feature_order():
    assert schema.to_csv([GOOD]).split(",") == [repr(float(GOOD[f])) for f in FEATURES]


# ---- drift ----------------------------------------------------------------------------------------------------

def test_psi_small_for_same_distribution_large_for_shift():
    rng = random.Random(0)
    ref = drift.reference({"x": [rng.gauss(0, 1) for _ in range(20_000)]})
    same = [rng.gauss(0, 1) for _ in range(2000)]
    moved = [rng.gauss(1, 1) for _ in range(2000)]
    assert abs(sum(ref["x"]["shares"]) - 1) < 1e-9
    assert drift.psi(ref["x"], same) < 0.02 < drift.THRESHOLD < drift.psi(ref["x"], moved)


def test_verdict_waits_for_enough_rows():
    assert drift.verdict({"x": 5.0}, drift.MIN_ROWS - 1)["status"] == "warming up"
    assert drift.verdict({"x": 5.0}, drift.MIN_ROWS)["status"] == "drift"


# ---- who is calling -------------------------------------------------------------------------------------------

def test_caller_system_and_people():
    clients = {"abc": "partner-a"}
    assert service.caller({"client_id": "abc", "scope": "inference/predict"}, clients)["partner"] == "partner-a"
    assert service.caller({"client_id": "zzz"}, clients)["partner"] is None
    person = service.caller({"username": "u1", "cognito:groups": "[partner-b]"}, clients)
    assert person["partner"] == "partner-b" and person["kind"] == "user" and not person["admin"]
    assert service.caller({"username": "u2", "cognito:groups": "[admin]"}, clients)["admin"]


def test_usage_is_isolated_by_partner():
    summary = {"partner-a": {"requests_today": 3}, "partner-b": {"requests_today": 9}}
    a = json.loads(service.usage({"partner": "partner-a", "admin": False}, summary)["body"])
    assert a == {"partner-a": {"requests_today": 3}}
    admin = json.loads(service.usage({"partner": None, "admin": True}, summary)["body"])
    assert set(admin) == {"partner-a", "partner-b"}
    assert service.usage({"partner": None, "admin": False}, summary)["statusCode"] == 403


def test_unmapped_caller_is_refused_before_the_model():
    called = []
    r = service.predict({"rows": [GOOD]}, {"partner": None}, lambda c: called.append(c), lambda r: None, "id")
    assert r["statusCode"] == 403 and not called


# ---- the whole demo, offline ----------------------------------------------------------------------------------

def test_local_demo_all_steps_pass(trained):
    from pgw import demo
    results = demo.local(verbose=False)
    assert [r["step"] for r in results] == list(range(1, 9))
    assert all(r["ok"] for r in results), results


# ---- the Lambda handlers, with fake AWS clients ---------------------------------------------------------------

class FakeS3:
    def __init__(self):
        self.objects = {}

        class _E:
            class NoSuchKey(Exception):
                pass
        self.exceptions = _E

    def put_object(self, Bucket, Key, Body, ContentType=None):
        self.objects[Key] = Body

    def get_object(self, Bucket, Key):
        if Key not in self.objects:
            raise self.exceptions.NoSuchKey()
        return {"Body": io.BytesIO(self.objects[Key])}

    def get_paginator(self, name):
        s3 = self

        class P:
            def paginate(self, Bucket, Prefix):
                yield {"Contents": [{"Key": k} for k in sorted(s3.objects) if k.startswith(Prefix)]}
        return P()


class FakeSM:
    def invoke_endpoint(self, EndpointName, ContentType, Accept, Body):
        n = len(Body.strip().splitlines())
        return {"Body": io.BytesIO(("\n".join(["2.5"] * n)).encode())}


class Ctx:
    aws_request_id = "req-1"


@pytest.fixture
def handlers(monkeypatch, trained):
    for k, v in {"AWS_DEFAULT_REGION": "us-east-1", "AWS_ACCESS_KEY_ID": "test", "AWS_SECRET_ACCESS_KEY": "test"}.items():
        monkeypatch.setenv(k, v)                    # fake credentials: these tests never reach AWS
    from pgw.aws import handlers as h
    monkeypatch.setattr(h, "_s3", FakeS3())
    monkeypatch.setattr(h, "_sm", FakeSM())
    monkeypatch.setenv("ENDPOINT", "ep")
    monkeypatch.setenv("BUCKET", "b")
    monkeypatch.setenv("CLIENTS", json.dumps({"cid-a": "partner-a", "cid-b": "partner-b"}))
    monkeypatch.setenv("TOPIC_ARN", "arn:topic")
    return h


def _event(claims, body):
    return {"requestContext": {"authorizer": {"claims": claims}}, "body": json.dumps(body)}


def test_predict_handler_logs_under_the_partner(handlers):
    r = handlers.predict(_event({"client_id": "cid-a"}, {"rows": [GOOD, GOOD]}), Ctx())
    assert r["statusCode"] == 200
    assert json.loads(r["body"])["predictions"] == [{"price_usd": 250_000}] * 2
    keys = list(handlers._s3.objects)
    assert len(keys) == 1 and keys[0].startswith("logs/partner-a/")


def test_drift_job_alerts_once_on_a_new_drift(handlers, monkeypatch):
    published = []

    class SNS:
        def publish(self, **kw):
            published.append(kw)
    monkeypatch.setattr(handlers.boto3, "client", lambda name: SNS())
    shifted = {**GOOD, "MedInc": 0.42}
    for i in range(3):
        handlers.predict(_event({"client_id": "cid-b"}, {"rows": [shifted] * 50}), Ctx())
    out = handlers.drift_job({}, None)
    assert out["partner-b"]["status"] == "drift" and out["partner-a"]["status"] == "warming up"
    assert len(published) == 1
    assert published[0]["MessageAttributes"]["partner"]["StringValue"] == "partner-b"
    handlers.drift_job({}, None)                     # still drifting: no second email
    assert len(published) == 1
