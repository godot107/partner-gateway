#!/usr/bin/env bash
# Once: a $5/month budget alert on project=partner-gateway. Kept when the other stacks are torn down.
set -euo pipefail
source "$(dirname "$0")/_common.sh"
require_iam_user
: "${ALERT_EMAIL:?set ALERT_EMAIL to the address for budget alerts}"
# shellcheck disable=SC2046
aws cloudformation deploy --stack-name "$COST_STACK" --template-file "$ROOT/infra/cost.yaml" \
  --no-fail-on-empty-changeset --tags $(tags iam) --parameter-overrides AlertEmail="$ALERT_EMAIL"
echo "Budget stack $COST_STACK ready."
