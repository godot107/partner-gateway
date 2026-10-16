# partner-gateway

Trello: Project Checklist items 10 (API Gateway to a SageMaker endpoint) and 12 (drift detection), as their own
card. **A demo of opening an ML model to outside partners**: Cognito (partner systems via client credentials,
partner staff via invitation-only managed login), API Gateway REST API (scopes, per-partner API keys + usage plans,
request schema), a thin Lambda in front of a SageMaker serverless endpoint, per-partner request logs and drift alerts.

## Run

```bash
.venv/bin/pip install -r requirements-dev.txt        # own venv: xgboost pinned to 1.7.6 (= container 1.7-1)
PYTHONPATH=src .venv/bin/python -m pytest -q         # 20 tests, no AWS (handlers tested with fake clients)
PYTHONPATH=src .venv/bin/python -m pgw train | demo-local | experiment | demo-aws
deploy/deploy_cost.sh (once) ; deploy/deploy.sh ; deploy/down.sh [--all]     # need `aws login` as WillieMaize
```

## Key decisions

- **Two routes.** `/predict` for partner systems needs a token with `inference/predict` AND the partner's API key
  (usage plan = rate/burst/daily quota, per-partner revocation by disabling the key). `/me/predict` and `/me/usage`
  for people need only the token; the Cognito group decides the partner. HTTP APIs can't do API keys or usage
  plans, so it's a REST API.
- **Thin Lambda, not a direct SageMaker integration** (pet-cnn uses the direct one): each request must be logged
  under its partner for drift monitoring, and serverless endpoints can't capture data themselves (verified). The Lambda code is pure Python + boto3: `service.py` and `drift.py` are shared with the local gateway,
  so `demo-local` exercises the same logic.
- **Model:** SageMaker built-in XGBoost container, model saved in XGBoost's binary format (`xgboost-model`), no
  pickle, no custom inference code. Trained on plain arrays (no feature names) because the container builds an
  unnamed DMatrix from CSV. Column order = `config.FEATURES`.
- **The contract** (`schema.REQUEST`) is the single source; `infra/app.yaml`'s PredictModel must equal it (tested).
  Ranges are wide on purpose: income in the wrong units still passes, which is what step 7 demonstrates.
- **Drift:** PSI over each partner's last 300 rows vs the training deciles, alarm > 0.25 once 100 rows are in; one
  email per new drift (SNS filter policy per partner; admin gets all).
- **Addresses:** real demo addresses (willieman.com, the user's choice) live only in gitignored `partners.yaml`
  and in AWS. Cognito built-in email sends invitations (user's choice; SES from willieman.com is a later option).
- **Verified 2026-10-16:** `XgbImage` URI (683313688378 / sagemaker-xgboost:1.7-1 in us-east-1, per the SageMaker
  SDK's image_uri_config); serverless endpoints do NOT support data capture (AWS docs exclusion list), so the thin
  Lambda logging is required; Cognito dropped the per-M2M-app-client fee in Nov 2025, only successful token requests
  are billed (~$2.25 per 1,000 at the first tier, third-party figure). **Still unverified:** 401 vs 403 for a missing
  scope on a REST API Cognito authorizer (HTTP APIs answer 403); the demo prints what it gets.
- **API Gateway redeploys:** CloudFormation doesn't redeploy an existing REST API when methods change;
  deploy.sh runs `create-deployment` after every stack update.
- **Tags:** shared/TAGGING.md, `project=partner-gateway`. CloudBurn scan in CI; one medium finding (S3
  storage-class transitions) accepted, as in pet-cnn: the bucket holds kilobytes and objects expire.
- **PUBLIC repo** godot107/partner-gateway (user's call, addresses kept out).
