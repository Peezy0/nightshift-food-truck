#!/usr/bin/env python3
"""Food Truck Wars bot. Python 3.8+, standard library only."""

import argparse
import datetime
import json
import os
from pathlib import Path
import sys
import time
import urllib.error
import urllib.request


API = "https://nightshift.joinbuilders.org/api/v1"
QUALITY_COST = {"cheap": 2.5, "standard": 3.75, "premium": 5.5}
SPOTS = ("oval", "thompson", "union", "stadium", "highst", "northres")


class EventLog:
    """Append small, key-free JSON records without interrupting an order."""

    def __init__(self, path):
        self.path = Path(path)
        self.run_id = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + str(os.getpid())
        self.warned = False

    def write(self, event, **fields):
        record = {"at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  "run_id": self.run_id, "event": event}
        record.update(fields)
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as output:
                output.write(json.dumps(record, separators=(",", ":"), default=str) + "\n")
        except OSError as error:
            if not self.warned:
                print("Logging unavailable: {}".format(error), file=sys.stderr, flush=True)
                self.warned = True

    def save_state(self, state):
        """Keep a readable latest-state snapshot beside the event log."""
        you = state.get("you") or {}
        last = state.get("last_day") or {}
        snapshot = {
            "observed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "version": state.get("version"),
            "mode": state.get("mode"),
            "market": state.get("market"),
            "markets": state.get("markets"),
            "day": state.get("day"),
            "days": state.get("days"),
            "phase": state.get("phase"),
            "phase_ends_at": state.get("phase_ends_at"),
            "server_time": state.get("server_time"),
            "orders_open": state.get("orders_open"),
            "order_for_day": state.get("order_for_day"),
            "rules": state.get("rules"),
            "forecast": state.get("forecast"),
            "you": {field: you.get(field) for field in
                    ("id", "truck_name", "company_name", "cash", "reputation", "order")},
            "last_day": {field: last.get(field) for field in
                         ("day", "weather", "events", "spots", "you")},
            "trucks": [{field: truck.get(field) for field in
                        ("id", "truck_name", "company_name", "color", "cash", "reputation_avg")}
                       for truck in state.get("trucks") or []],
            "announcements": state.get("announcements") or [],
        }
        destination = self.path.parent / "latest-state.json"
        temporary = self.path.parent / "latest-state.json.tmp"
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary.write_text(json.dumps(snapshot, indent=2, default=str) + "\n", encoding="utf-8")
            temporary.replace(destination)
        except OSError as error:
            if not self.warned:
                print("Status snapshot unavailable: {}".format(error), file=sys.stderr, flush=True)
                self.warned = True


def log_decision(log, state, order):
    spot = order["spot"]
    previous = ((state.get("last_day") or {}).get("spots") or {}).get(spot) or {}
    rivals = [truck for truck in previous.get("trucks") or []
              if truck.get("id") != (state.get("you") or {}).get("id")]
    forecast = state.get("forecast") or {}
    log.write("decision", mode=state.get("mode"), market=state.get("market"),
              day=state.get("order_for_day"), version=state.get("version"),
              order=order, expected_customers=(forecast.get("expected_customers") or {}).get(spot),
              crowd=(forecast.get("crowds") or {}).get(spot),
              weather=forecast.get("weather"), events=forecast.get("events"),
              reputation=((state.get("you") or {}).get("reputation") or {}).get(spot),
              recent_rivals=len(rivals), recent_rival_prices=[truck.get("price") for truck in rivals])


def log_result(log, state):
    last = state.get("last_day") or {}
    if last.get("day") is None:
        return
    own = last.get("you") or {}
    spot = own.get("spot")
    spot_data = (last.get("spots") or {}).get(spot) or {}
    meals = int(own.get("meals") or 0)
    sold = int(own.get("sold") or 0)
    log.write("result", mode=state.get("mode"), market=state.get("market"),
              day=last.get("day"), spot=spot, price=own.get("price"),
              quality=own.get("quality"), meals=meals, sold=sold,
              spoiled=max(0, meals - sold), revenue=own.get("revenue"),
              cost=own.get("cost"), profit=own.get("profit"),
              cash=(state.get("you") or {}).get("cash"), weather=last.get("weather"),
              customers=spot_data.get("customers"), hungry=spot_data.get("hungry"),
              rival_count=max(0, len(spot_data.get("trucks") or []) - (1 if spot else 0)),
              missed=not bool(own.get("spot")))


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
    rules = state.get("rules") or {}
    spots = rules.get("spots") or SPOTS

    best_spot = spots[0]
    best_score = float("-inf")
    for spot in spots:
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
    price = max(float(rules.get("price_min", 1)), min(9.5, float(rules.get("price_max", 30))))
    # Unsold meals spoil. Estimate our share, then leave a modest buffer.
    recent_rivals = sum(1 for truck in (previous.get(best_spot) or {}).get("trucks") or []
                        if truck.get("id") != (state.get("you") or {}).get("id"))
    likely_sales = float(expected.get(best_spot) or 60) / (1 + 1.2 * recent_rivals)
    cost = float((rules.get("costs") or {}).get(quality, QUALITY_COST[quality]))
    budget = float(rules.get("budget", 500))
    meals = min(int(budget // cost), max(15, round(likely_sales * 0.75)))
    return {"spot": best_spot, "price": price, "quality": quality, "meals": meals}


def practice(key, days, rivals, seed, log):
    options = {"days": days, "rivals": rivals}
    if seed:
        options["seed"] = seed
    game = request("POST", API + "/practice", key, options, timeout=10)
    base = "https://nightshift.joinbuilders.org" + game["base"]
    state = request("GET", base + "/state", key)
    while state.get("phase") != "over":
        order = decide(state)
        log_decision(log, state, order)
        result = request("POST", base + "/orders", key, order, timeout=10)
        state = result.get("state") or request("GET", base + "/state", key)
        log_result(log, state)
        result_line = (state.get("last_day") or {}).get("you") or {}
        print("Day {}: {} | profit ${:.2f} | cash ${:.2f}".format(
            state.get("day"), order, float(result_line.get("profit") or 0),
            float((state.get("you") or {}).get("cash") or 0)), flush=True)
    print("Practice complete. Final cash: ${:.2f}".format(
        float((state.get("you") or {}).get("cash") or 0)), flush=True)


def live(key, log):
    version = 0
    seen_results = set()
    connected = False
    while True:
        try:
            url = API + "/state"
            if version:
                url += "?since=" + str(version)
            state = request("GET", url, key, timeout=30)
            version = state.get("version") or 0
            if not connected:
                print("Connected: {} market {} day {} ({})".format(
                    state.get("mode"), state.get("market"), state.get("day"),
                    state.get("phase")), flush=True)
                connected = True
            last = state.get("last_day") or {}
            result_id = (state.get("mode"), state.get("market"), last.get("day"))
            if last.get("day") is not None and result_id not in seen_results:
                log_result(log, state)
                seen_results.add(result_id)
            if state.get("phase") == "night" and state.get("orders_open"):
                day = state.get("order_for_day")
                existing = (state.get("you") or {}).get("order")
                if not existing:
                    order = decide(state)
                    log_decision(log, state, order)
                    started = time.monotonic()
                    response = request("POST", API + "/orders", key, order, timeout=5)
                    elapsed_ms = round((time.monotonic() - started) * 1000)
                    log.write("order_submitted", mode=state.get("mode"), market=state.get("market"),
                              day=day, order=order, latency_ms=elapsed_ms)
                    (state.get("you") or {})["order"] = response.get("order") or order
                    print("{} market {} day {}: submitted {}".format(
                        state.get("mode"), state.get("market"), day, order), flush=True)
            log.save_state(state)
        except KeyboardInterrupt:
            print("Stopping bot.", flush=True)
            return
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            log.write("error", status=error.code, error_type="HTTPError")
            print("HTTP {}: {}".format(error.code, detail), file=sys.stderr, flush=True)
            if error.code == 401:
                return
            time.sleep(0.5 if error.code == 429 else 0.2)
        except (OSError, ValueError, KeyError) as error:
            log.write("error", error_type=type(error).__name__)
            print("Connection/state error: {}".format(error), file=sys.stderr, flush=True)
            time.sleep(0.2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--practice", action="store_true", help="run a private practice market")
    parser.add_argument("--days", type=int, default=30, help="practice days")
    parser.add_argument("--rivals", type=int, default=3, help="practice house bots")
    parser.add_argument("--seed", default=None, help="repeatable practice seed")
    parser.add_argument("--log", default=None, help="path to JSONL event log")
    args = parser.parse_args()
    key = os.environ.get("NIGHTSHIFT_KEY")
    if not key:
        parser.error("Set NIGHTSHIFT_KEY to your truck key before running the bot")
    log_path = args.log
    if log_path is None:
        folder = Path(__file__).resolve().parent / "logs"
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        log_path = folder / ("practice-{}-{}.jsonl".format(stamp, os.getpid()) if args.practice else "live.jsonl")
    log = EventLog(log_path)
    log.write("run_started", mode="practice" if args.practice else "live", log_path=str(log.path))
    if args.practice:
        practice(key, args.days, args.rivals, args.seed, log)
    else:
        live(key, log)


if __name__ == "__main__":
    main()
