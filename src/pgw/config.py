"""Single source of truth: paths, the model's features, and the partner configuration."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
MODELS = ROOT / "models"
RESULTS = ROOT / "reports" / "results"
FIGURES = ROOT / "reports" / "figures"
SEED = 42

# California housing (1990 census block groups). Order matters: the endpoint takes CSV rows in this order.
FEATURES = ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup", "Latitude", "Longitude"]
TARGET_UNIT = 100_000          # the target is median house value in units of $100,000

PARTNERS = ("partner-a", "partner-b")   # fixed in infra/app.yaml (one app client + usage plan each)
SCOPES = {"predict": "inference/predict", "usage": "inference/usage"}


def partners_file() -> Path:
    """partners.yaml (gitignored, real addresses) if present, else the example with placeholders."""
    real = ROOT / "partners.yaml"
    return real if real.exists() else ROOT / "partners.example.yaml"


def load_partners() -> dict:
    import yaml
    cfg = yaml.safe_load(partners_file().read_text())
    missing = set(PARTNERS) - set(cfg["partners"])
    if missing:
        raise ValueError(f"{partners_file().name} lacks partners {sorted(missing)}")
    return cfg
