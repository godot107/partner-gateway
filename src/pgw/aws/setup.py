"""Post-deploy setup from partners.yaml (run by deploy/deploy.sh with the operator's credentials). Idempotent.

  * Cognito users: one per address, invited by Cognito's built-in email (a temporary password), added to their
    partner's group; the admin address goes in the "admin" group.
  * SNS email subscriptions to the drift topic: each partner's addresses filtered to that partner's alerts; the
    admin address gets every alert. Each address must click the confirmation link SNS sends.

The real addresses live only in the gitignored partners.yaml and in AWS, never in the repo or in stack parameters.
"""
from __future__ import annotations

import json
import sys

import boto3

from ..config import ROOT, load_partners
from .client import STACK


def outputs(stack=STACK):
    s = boto3.client("cloudformation").describe_stacks(StackName=stack)["Stacks"][0]
    return {o["OutputKey"]: o["OutputValue"] for o in s["Outputs"]}


def ensure_user(cog, pool, email, group):
    try:
        cog.admin_get_user(UserPoolId=pool, Username=email)
        print(f"  user exists: {email}")
    except cog.exceptions.UserNotFoundException:
        cog.admin_create_user(UserPoolId=pool, Username=email, DesiredDeliveryMediums=["EMAIL"],
                              UserAttributes=[{"Name": "email", "Value": email},
                                              {"Name": "email_verified", "Value": "true"}])
        print(f"  invited: {email} (temporary password sent by Cognito)")
    cog.admin_add_user_to_group(UserPoolId=pool, Username=email, GroupName=group)


def ensure_subscription(sns, topic, email, partner=None):
    subs = [s for page in sns.get_paginator("list_subscriptions_by_topic").paginate(TopicArn=topic)
            for s in page["Subscriptions"]]
    if any(s["Endpoint"] == email for s in subs):
        print(f"  alert subscription exists: {email}")
        return
    attrs = {"FilterPolicy": json.dumps({"partner": [partner]})} if partner else {}
    sns.subscribe(TopicArn=topic, Protocol="email", Endpoint=email, Attributes=attrs)
    print(f"  alert subscription requested: {email} -> {partner or 'all partners'} (confirm the email)")


def main():
    if not (ROOT / "partners.yaml").exists():
        sys.exit("partners.yaml not found: copy partners.example.yaml and put the real addresses in it.")
    cfg, o = load_partners(), outputs()
    cog, sns = boto3.client("cognito-idp"), boto3.client("sns")
    print("Cognito users and groups:")
    ensure_user(cog, o["UserPoolId"], cfg["admin_email"], "admin")
    for partner, p in cfg["partners"].items():
        for email in p["users"]:
            ensure_user(cog, o["UserPoolId"], email, partner)
    print("Drift alert subscriptions:")
    ensure_subscription(sns, o["TopicArn"], cfg["admin_email"])
    for partner, p in cfg["partners"].items():
        for email in p.get("alert_emails", []):
            ensure_subscription(sns, o["TopicArn"], email, partner)
    print(f"\nPartner staff sign in at: {o['WebPage']}")


if __name__ == "__main__":
    main()
