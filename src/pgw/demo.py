"""The eight-step demo, against the local stand-in (`--local`) or the deployed stack (`--aws`).

  1  no token                                   -> 401
  2  token without the predict scope            -> 403 (AWS may say 401)
  3  Partner A, normal traffic                  -> 200 + prices; also a normal day of traffic for the monitor
  4  malformed request                          -> 400, rejected before the model runs
  5  Partner B bursts 12 requests at once       -> some 429 (its usage plan allows 1/s, burst 2)
  6  Partner B's API key disabled               -> 403 for B; A still 200; then re-enabled
  7  Partner B sends income in $100k units       -> still 200, but the prices are badly wrong
  8  drift check                                -> Partner B flagged, Partner A ok
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from .config import FEATURES, MODELS, SCOPES, load_partners

BAD_ROW = {"MedInc": "high", "HouseAge": 20}                    # wrong type, missing fields


def rows(df, n, seed):
    return df.sample(n, random_state=seed)


def body(df):
    return {"rows": df[FEATURES].to_dict("records")}


def run(gw, verbose=True) -> list[dict]:
    pool = pd.read_csv(MODELS / "partner_pool.csv")
    a_id, a_secret, a_key = gw.credentials("partner-a")
    b_id, b_secret, b_key = gw.credentials("partner-b")
    _, a_tok = gw.token(a_id, a_secret)
    _, b_tok = gw.token(b_id, b_secret)
    a_tok, b_tok = a_tok["access_token"], b_tok["access_token"]
    out = []

    def step(n, title, expect, got, note=""):
        ok = got in expect if isinstance(expect, set) else got == expect      # a set = any of these
        out.append({"step": n, "title": title, "expected": expect, "got": got, "ok": ok, "note": note})
        if verbose:
            print(f"{'PASS' if ok else 'CHECK'}  {n}. {title}: expected {expect}, got {got}  {note}")

    s, _ = gw.predict(body(rows(pool, 1, 1)), token=None, api_key=a_key)
    step(1, "no token", 401, s)

    _, t = gw.token(a_id, a_secret, scope=SCOPES["usage"])
    s, _ = gw.predict(body(rows(pool, 1, 2)), token=t["access_token"], api_key=a_key)
    step(2, "token without the predict scope", {401, 403}, s)

    one = rows(pool, 1, 3)
    s, r = gw.predict(body(one), token=a_tok, api_key=a_key)
    price = r["predictions"][0]["price_usd"] if s == 200 else None
    step(3, "Partner A, one house", 200, s,
         f"predicted ${price:,}, actual ${one.price.iloc[0] * 100_000:,.0f}" if price else "")
    for i in range(3):                                        # a normal day of traffic: 300 rows
        gw.predict(body(rows(pool, 100, 100 + i)), token=a_tok, api_key=a_key)

    s, r = gw.predict({"rows": [BAD_ROW]}, token=a_tok, api_key=a_key)
    step(4, "malformed request", 400, s, "never reached the model")

    with ThreadPoolExecutor(12) as ex:
        codes = list(ex.map(lambda i: gw.predict(body(rows(pool, 1, 200 + i)), token=b_tok, api_key=b_key)[0],
                            range(12)))
    step(5, "Partner B bursts 12 requests", True, 429 in codes,
         f"{codes.count(200)} x 200, {codes.count(429)} x 429")
    time.sleep(3)                                              # let B's bucket refill

    gw.set_key_enabled("partner-b", False)
    sb, _ = gw.predict(body(rows(pool, 1, 4)), token=b_tok, api_key=b_key)
    sa, _ = gw.predict(body(rows(pool, 1, 5)), token=a_tok, api_key=a_key)
    gw.set_key_enabled("partner-b", True)
    step(6, "Partner B's key disabled (A unaffected)", (403, 200), (sb, sa))
    time.sleep(1)

    codes, errs = [], []
    for i in range(3):                                          # 300 rows with income in the wrong units
        d = rows(pool, 100, 300 + i)
        sent = d.assign(MedInc=d.MedInc * 0.1)
        s, r = gw.predict(body(sent), token=b_tok, api_key=b_key)
        codes.append(s)
        if s == 200:
            p = pd.Series([x["price_usd"] for x in r["predictions"]])
            errs.extend((p - d.price.to_numpy() * 100_000).abs())
        time.sleep(1.1)
    mae = sum(errs) / len(errs) if errs else float("nan")
    step(7, "Partner B sends income in $100k units", True, all(c == 200 for c in codes),
         f"all 200s; mean error ${mae:,.0f} per house")

    v = gw.drift_check()
    step(8, "drift check", ("drift", "ok"), (v["partner-b"]["status"], v["partner-a"]["status"]),
         f"B worst: {v['partner-b']['worst_feature']} PSI {v['partner-b']['worst_psi']:.2f}; "
         f"A worst: {v['partner-a']['worst_feature']} PSI {v['partner-a']['worst_psi']:.2f}")
    return out


def local(verbose=True):
    from .local import LocalGateway
    return run(LocalGateway(load_partners()), verbose)
