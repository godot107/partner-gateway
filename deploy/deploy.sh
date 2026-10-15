#!/usr/bin/env bash
# Everything for the demo: bucket, model artifact, Lambda code, the app stack, then users and alert emails from
# partners.yaml. Safe to re-run. Tear down with deploy/down.sh.
set -euo pipefail
source "$(dirname "$0")/_common.sh"
require_iam_user
[ -f "$ROOT/partners.yaml" ] || { echo "Copy partners.example.yaml to partners.yaml and fill in the addresses." >&2; exit 1; }

# shellcheck disable=SC2046
aws cloudformation deploy --stack-name "$DATA_STACK" --template-file "$ROOT/infra/data.yaml" \
  --no-fail-on-empty-changeset --tags $(tags data)
BUCKET=$(out "$DATA_STACK" BucketName)

[ -f "$ROOT/models/model.tar.gz" ] || "$PY" -m pgw train
aws s3 cp "$ROOT/models/model.tar.gz" "s3://$BUCKET/artifacts/model.tar.gz" --only-show-errors

# Lambda package: the pure-Python parts of pgw (no numpy, no xgboost) + the drift reference + the web page.
BUILD="$ROOT/build/lambda"
rm -rf "$BUILD" && mkdir -p "$BUILD/pgw/aws"
cp "$ROOT"/src/pgw/{__init__,config,drift,schema,service}.py "$BUILD/pgw/"
cp "$ROOT"/src/pgw/aws/{__init__,handlers}.py "$BUILD/pgw/aws/"
cp "$ROOT/models/reference.json" "$ROOT/web/index.html" "$BUILD/pgw/"
(cd "$BUILD" && find . -exec touch -t 202001010000 {} + && zip -qrX ../lambda.zip pgw)
HASH=$(sha256sum "$ROOT/build/lambda.zip" | cut -c1-12)
CODE_KEY="code/lambda-$HASH.zip"
aws s3 cp "$ROOT/build/lambda.zip" "s3://$BUCKET/$CODE_KEY" --only-show-errors

ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
read -r RA BA QA RB BB QB < <("$PY" -c "
from pgw.config import load_partners
p = load_partners()['partners']
print(*[p[k][f] for k in ('partner-a', 'partner-b') for f in ('rate_per_s', 'burst', 'quota_per_day')])")

# shellcheck disable=SC2046
aws cloudformation deploy --stack-name "$APP_STACK" --template-file "$ROOT/infra/app.yaml" \
  --capabilities CAPABILITY_IAM --no-fail-on-empty-changeset --tags $(tags app) \
  --parameter-overrides BucketName="$BUCKET" CodeKey="$CODE_KEY" DomainPrefix="pgw-demo-$ACCOUNT" \
    RateA="$RA" BurstA="$BA" QuotaA="$QA" RateB="$RB" BurstB="$BB" QuotaB="$QB" \
    ${XGB_IMAGE:+XgbImage="$XGB_IMAGE"}

# CloudFormation doesn't redeploy an existing REST API when its methods change: publish the current config.
aws apigateway create-deployment --rest-api-id "$(out "$APP_STACK" ApiId)" --stage-name v1 \
  --description "deploy.sh $(date -u +%FT%TZ)" >/dev/null

"$PY" -m pgw.aws.setup
echo
echo "Web page:  $(out "$APP_STACK" WebPage)"
echo "Dashboard: https://$AWS_REGION.console.aws.amazon.com/cloudwatch/home?region=$AWS_REGION#dashboards/dashboard/partner-gateway"
echo "Demo:      PYTHONPATH=src .venv/bin/python -m pgw demo-aws"
