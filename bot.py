#!/usr/bin/env python3
"""Food Truck Wars bot. Python 3.8+, standard library only."""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request


API = "https://nightshift.joinbuilders.org/api/v1"
QUALITY_COST = {"cheap": 2.5, "standard": 3.75, "premium": 5.5}
SPOTS = ("oval", "thompson", "union", "stadium", "highst", "northres")


def request(method, url, key, payload=None, timeout=30):
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Authorization": "Bearer " + key}
    if body is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def decide(state):
    """Pick a spot with demand and little recent competition."""
    forecast = state.get("forecast") or {}
    expected = forecast.get("expected_customers") or {}
    previous = (state.get("last_day") or {}).get("spots") or {}
    reputation = (state.get("you") or {}).get("reputation") or {}
    events = forecast.get("events") or []

    best_spot = "union"
    best_score = float("-inf")
    for spot in SPOTS:
        customers = float(expected.get(spot) or 0)
        recent_trucks = (previous.get(spot) or {}).get("trucks") or []
        # Last day's rivals are only a hint; they may move tomorrow.
        rivals = sum(1 for truck in recent_trucks if truck.get("id") != (state.get("you") or {}).get("id"))
        loyal = float(reputation.get(spot, 50))
        event_bonus = 1.08 if any(event.get("spot") == spot for event in events) else 1.0
        score = customers * event_bonus * (0.75 + loyal / 200) / (1 + 0.7 * rivals)
        if score > best_score:
            best_score, best_spot = score, spot

    quality = "standard"
    price = 9.5
    # Unsold meals spoil. Estimate our share, then leave a modest buffer.
    recent_rivals = sum(1 for truck in (previous.get(best_spot) or {}).get("trucks") or []
                        if truck.get("id") != (state.get("you") or {}).get("id"))
    likely_sales = float(expected.get(best_spot) or 60) / (1 + 1.2 * recent_rivals)
    meals = min(int(500 // QUALITY_COST[quality]), max(15, round(likely_sales * 0.75)))
    return {"spot": best_spot, "price": price, "quality": quality, "meals": meals}


def practice(key, days, rivals, seed):
    options = {"days": days, "rivals": rivals}
    if seed:
        options["seed"] = seed
    game = request("POST", API + "/practice", key, options, timeout=10)
    base = "https://nightshift.joinbuilders.org" + game["base"]
    state = request("GET", base + "/state", key)
    while state.get("phase") != "over":
        order = decide(state)
        result = request("POST", base + "/orders", key, order, timeout=10)
        state = result.get("state") or request("GET", base + "/state", key)
        result_line = (state.get("last_day") or {}).get("you") or {}
        print("Day {}: {} | profit ${:.2f} | cash ${:.2f}".format(
            state.get("day"), order, float(result_line.get("profit") or 0),
            float((state.get("you") or {}).get("cash") or 0)), flush=True)
    print("Practice complete. Final cash: ${:.2f}".format(
        float((state.get("you") or {}).get("cash") or 0)), flush=True)


def live(key):
    version = 0
    while True:
        try:
            url = API + "/state"
            if version:
                url += "?since=" + str(version)
            state = request("GET", url, key, timeout=30)
            version = state.get("version") or 0
            if state.get("phase") != "night" or not state.get("orders_open"):
                continue
            day = state.get("order_for_day")
            existing = (state.get("you") or {}).get("order")
            if existing:
                continue
            order = decide(state)
            request("POST", API + "/orders", key, order, timeout=5)
            print("{} market {} day {}: submitted {}".format(
                state.get("mode"), state.get("market"), day, order), flush=True)
        except KeyboardInterrupt:
            print("Stopping bot.", flush=True)
            return
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            print("HTTP {}: {}".format(error.code, detail), file=sys.stderr, flush=True)
            if error.code == 401:
                return
            time.sleep(0.5 if error.code == 429 else 0.2)
        except (OSError, ValueError, KeyError) as error:
            print("Connection/state error: {}".format(error), file=sys.stderr, flush=True)
            time.sleep(0.2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--practice", action="store_true", help="run a private practice market")
    parser.add_argument("--days", type=int, default=30, help="practice days")
    parser.add_argument("--rivals", type=int, default=3, help="practice house bots")
    parser.add_argument("--seed", default=None, help="repeatable practice seed")
    args = parser.parse_args()
    key = os.environ.get("NIGHTSHIFT_KEY")
    if not key:
        parser.error("Set NIGHTSHIFT_KEY to your truck key before running the bot")
    if args.practice:
        practice(key, args.days, args.rivals, args.seed)
    else:
        live(key)


if __name__ == "__main__":
    main()
