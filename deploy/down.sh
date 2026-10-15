#!/usr/bin/env bash
# Delete the app stack (endpoint, API, Cognito users, functions). --all also empties and deletes the bucket.
# The budget stack is kept.
set -euo pipefail
source "$(dirname "$0")/_common.sh"
require_iam_user
aws cloudformation delete-stack --stack-name "$APP_STACK"
aws cloudformation wait stack-delete-complete --stack-name "$APP_STACK"
echo "Deleted $APP_STACK."
if [ "${1:-}" = "--all" ]; then
  BUCKET=$(out "$DATA_STACK" BucketName)
  aws s3 rm "s3://$BUCKET" --recursive --only-show-errors
  aws cloudformation delete-stack --stack-name "$DATA_STACK"
  aws cloudformation wait stack-delete-complete --stack-name "$DATA_STACK"
  echo "Deleted $DATA_STACK and its bucket."
fi
