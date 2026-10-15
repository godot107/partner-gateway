"""The request contract for POST /v1/predict. API Gateway validates against the same schema (infra/app.yaml;
a test checks the two are identical), so a malformed request is rejected with 400 before any code runs or the
model is billed.

The ranges are deliberately wide: a contract can say "a number between 0 and 20", but it can't tell a correct
income from one sent in the wrong units. That gap is what drift monitoring is for.
"""
from __future__ import annotations

from .config import FEATURES

RANGES = {"MedInc": (0, 20), "HouseAge": (0, 100), "AveRooms": (0, 200), "AveBedrms": (0, 50),
          "Population": (0, 50_000), "AveOccup": (0, 1_500), "Latitude": (32, 42.5), "Longitude": (-125, -114)}

ROW = {"type": "object", "additionalProperties": False, "required": FEATURES,
       "properties": {f: {"type": "number", "minimum": lo, "maximum": hi} for f, (lo, hi) in RANGES.items()}}

REQUEST = {"$schema": "http://json-schema.org/draft-04/schema#",           # API Gateway models use draft 4
           "title": "PredictRequest", "type": "object", "additionalProperties": False, "required": ["rows"],
           "properties": {"rows": {"type": "array", "minItems": 1, "maxItems": 100, "items": ROW}}}


def errors(body) -> list[str]:
    """Schema violations, as API Gateway would report them (empty list = valid)."""
    import jsonschema
    v = jsonschema.Draft4Validator(REQUEST)
    return [e.message for e in sorted(v.iter_errors(body), key=lambda e: list(e.path))]


def to_csv(rows: list[dict]) -> str:
    """The built-in XGBoost container takes headerless CSV in feature order."""
    return "\n".join(",".join(repr(float(r[f])) for f in FEATURES) for r in rows)


def quick_check(body) -> str | None:
    """The same contract in plain Python (the Lambda has no jsonschema): first problem found, or None.
    API Gateway already enforces REQUEST; this is defence in depth for direct invocations and the web page."""
    if not isinstance(body, dict) or set(body) != {"rows"} or not isinstance(body["rows"], list):
        return "body must be {\"rows\": [...]}"
    if not 1 <= len(body["rows"]) <= 100:
        return "rows must hold 1 to 100 items"
    for i, r in enumerate(body["rows"]):
        if not isinstance(r, dict) or set(r) != set(FEATURES):
            return f"row {i}: needs exactly {FEATURES}"
        for f, (lo, hi) in RANGES.items():
            v = r[f]
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not lo <= v <= hi:
                return f"row {i}: {f} must be a number in [{lo}, {hi}]"
    return None
