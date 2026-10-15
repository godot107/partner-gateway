# Shared settings for deploy scripts. Source, don't run.
DATA_STACK=${DATA_STACK:-partner-gateway-data}
APP_STACK=${APP_STACK:-partner-gateway-app}
COST_STACK=${COST_STACK:-partner-gateway-cost}
export AWS_REGION=${AWS_REGION:-us-east-1}
export AWS_DEFAULT_REGION=$AWS_REGION
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PY="$ROOT/.venv/bin/python"
export PYTHONPATH="$ROOT/src"

# shared/TAGGING.md: CloudFormation copies stack tags to every taggable resource.
tags() {
  echo project=partner-gateway environment=dev owner=williemaize cost-center=personal \
    managed-by=cloudformation component="$1" repository=github.com/godot107/partner-gateway
}

require_iam_user() {
  local arn
  arn=$(aws sts get-caller-identity --query Arn --output text) || {
    echo "No AWS session: run 'aws login' first." >&2; exit 1; }
  case "$arn" in
    *:root) echo "Refusing to deploy as the root user. 'aws login' as WillieMaize." >&2; exit 1 ;;
  esac
}

out() {  # out <stack> <OutputKey>
  aws cloudformation describe-stacks --stack-name "$1" \
    --query "Stacks[0].Outputs[?OutputKey=='$2'].OutputValue" --output text
}
