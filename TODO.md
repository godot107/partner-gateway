# TODO

## Done (offline)
- [x] Model (XGBoost, R² 0.85), drift reference, partner pool
- [x] Contract (schema), service, local gateway, eight-step demo: all pass offline
- [x] Drift experiment: detection speed and harm by kind and size of change
- [x] Infra: cost, data, app stacks (cfn-lint clean); deploy / down scripts; Lambda handlers tested with fakes

## Needs `aws login` (WillieMaize)
- [ ] Fill partners.yaml with the willieman.com demo addresses (gitignored)
- [ ] Verify the XGBoost image URI for us-east-1 and current Cognito pricing (M2M tokens) before deploying
- [ ] deploy_cost.sh, deploy.sh; confirm the SNS emails; sign in with each demo address
- [ ] `pgw demo-aws`; record what API Gateway returns for a missing scope (401 or 403); screenshot the web page and
      the dashboard; add results to README, blog and notebook
- [ ] down.sh --all after recording

## Ideas
- [ ] False-alarm vs detection-speed trade-off (warm-up length, threshold, window)
- [ ] Per-partner drift on the predictions alone (no raw inputs stored) for privacy-sensitive partners
- [ ] Custom domain (api.willieman.com) and SES sending from willieman.com
- [ ] Federate a partner's own IdP (SAML/OIDC) into the user pool
