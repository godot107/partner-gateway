# Your model works. Now a partner wants to call it.

*What it takes to open a machine-learning model to outside businesses, and the failure no security check can catch.*

Most machine-learning projects end with a model that works on someone's laptop. The interesting questions start
when someone *outside* wants to use it: a vendor, a partner, a customer. Who are they? How much may they use? What
happens when one of them misbehaves? And what happens when one of them *doesn't* misbehave, exactly, but quietly
starts sending data your model wasn't built for?

I built a small, complete version of this to find out. Code, tests and an executed notebook:
[github.com/godot107/partner-gateway](https://github.com/godot107/partner-gateway).

## The setup

The model prices homes (XGBoost on the classic California census data; on held-out data it explains 85% of the
price variation, with a typical error around $30,000). Two businesses get access: **Partner A**, a realtor, and
**Partner B**, a lender. Each has two ways in:

- **Their systems**, calling the API directly. Each partner gets credentials (a client ID and secret) that it trades
  for a short-lived access token, plus an API key that ties its traffic to its own quota.
- **Their staff**, signing in to a small web page with an invitation-only account. A person's group (`partner-a`,
  `partner-b` or `admin`) decides what they can see: their own partner's usage and data health, never the other's.

On AWS that's Amazon Cognito for identity, API Gateway in front, a small Lambda function, and the model on a
SageMaker serverless endpoint. Everything scales to zero when nobody's calling, and one script deletes it all.

## Eight things that should happen

The demo runs eight steps and checks each answer:

| Step | What happens | Answer |
|---|---|---|
| 1 | A call with no token | **401**: who are you? |
| 2 | A token without permission to predict | **403**: not allowed |
| 3 | Partner A prices a house | **200** and a price |
| 4 | A malformed request | **400**, before the model runs, so it costs nothing |
| 5 | Partner B fires 12 requests at once | mostly **429**: its plan allows one per second |
| 6 | Partner B's key is switched off | B gets **403**; A keeps working |
| 7 | Partner B starts sending income in the wrong units | **200**, every time |
| 8 | The drift check runs | B is flagged |

The first six are the parts everyone thinks of: authentication, permission, input checking, rate limits and the
ability to cut one partner off without touching the others. They all happen at the gateway, before any of my code
runs, which is the point: the cheapest request to reject is the one that never reaches the model.

One detail surprised me. Tokens can't really be revoked quickly; they stay valid until they expire (an hour here).
The per-partner API key is what makes step 6 instant. Belt and braces turned out to be the right design.

## Step 7 is the interesting one

Partner B's engineers change a field. Median household income used to be sent in tens of thousands of dollars
(4.2 meaning $42,000); now it's sent in hundreds of thousands (0.42). Every request is still well formed. 0.42 is a
perfectly plausible number; it just means something else. The schema accepts it, the gateway accepts it, the model
returns a price, and Partner B gets a steady stream of 200 OK responses.

The prices are now off by about $66,000 per house on average, more than double the model's normal error. Nothing
in the security stack can see this, because nothing is wrong *with the request*. Something is wrong with what it
means.

## Watching the data, not just the door

The fix is to keep an eye on what each partner sends. Every request is logged under its partner. Periodically, each
partner's last 300 houses are compared with the data the model was trained on, input by input, using the
*population stability index*: split the training values into ten equal-sized bins, see what share of the partner's
values land in each, and add up how far the shares have moved. Near zero means "looks like training"; above 0.25 is
the usual line for "shifted". When a partner crosses it, the partner's contact and the provider both get an email.

In step 8, Partner A's inputs all scored below 0.1. Partner B's income scored **8.3**, while every other input
stayed below 0.07, which points straight at the field to ask them about. B's predictions shifted too (0.37): the
consequence, not the cause.

## How fast, and at what cost?

Because the simulated partners send real houses the model never trained on, I know the true price of every request,
so I could measure both how quickly a change gets caught and how much damage it does in the meantime:

![Drift experiment](https://raw.githubusercontent.com/godot107/partner-gateway/master/reports/figures/drift_experiment.png)

- **Harmful changes are caught fast.** Income at a tenth or half of its real value was flagged within 15 to 20
  requests; a 25% error within 50. Meanwhile, the typical price error climbed from $30k to as much as $63k.
- **Harmless changes are ignored.** Income 10% low barely changes the error, and the monitor stays quiet. That's
  the behaviour you want.
- **Drift isn't damage.** When Partner B moved into a new market (the Bay Area), the monitor fired, but the model's
  error rose only 12%. An alert says "go look", not "the model is broken".
- **False alarms exist.** In 7.5% of runs, the monitor raised an alarm before anything changed. A longer warm-up or
  a higher threshold would cut that, at the cost of catching real problems later.

## What I'd tell a team opening a model to partners

1. **Give every partner two handles**: a token (who they are, what they may do) and a key (their quota, and a switch
   you can flip instantly).
2. **Reject bad requests at the gateway** with a schema. It's free there, and costs money and confusion further in.
3. **Keep partners apart**: separate quotas, separate logs, and staff who can only see their own company's data.
4. **Monitor each partner's inputs, not just errors and latency.** The worst failures return 200 OK.
5. **Treat a drift alert as a conversation starter.** Sometimes the partner broke something; sometimes their
   business changed and the model is fine; sometimes the model needs retraining. The alert can't tell which.

The demo runs entirely offline today (a faithful stand-in for Cognito and API Gateway runs the same code the AWS
Lambda runs). The AWS deployment is scripted; I'll add screenshots of the live run and the sign-in page once it's up.
