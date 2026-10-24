#!/usr/bin/env python3
"""Food Truck Wars bot. Python 3.8+, standard library only."""

import argparse
import datetime
import importlib
import json
import math
import os
import shutil
from pathlib import Path
import statistics
import sys
import time
import urllib.error
import urllib.request

import agent as profit_agent
from agent import Agent


API = "https://nightshift.joinbuilders.org/api/v1"
SETTINGS = Path(__file__).resolve().parent / "agent-settings.json"
AGENT_SOURCE = Path(profit_agent.__file__).resolve()
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
            "features": state.get("features"),
            "forecast": state.get("forecast"),
            "spot_info": state.get("spot_info"),
            "you": {field: you.get(field) for field in
                    ("id", "truck_name", "company_name", "cash", "reputation", "order", "last_spot", "staff",
                     "coupons", "coupons_dropped", "wrecked_until_day", "vote", "news")},
            "last_day": {field: last.get(field) for field in
                         ("day", "weather", "events", "spots", "you", "moments", "labor")},
            "trucks": [{field: truck.get(field) for field in
                        ("id", "truck_name", "company_name", "color", "cash", "reputation_avg", "employees",
                         "wage", "badge")}
                       for truck in state.get("trucks") or []],
            "labor": state.get("labor"),
            "politics": state.get("politics"),
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


def valid_order(order, rules):
    """True when an order obeys the published limits, so a bad plan is never submitted."""
    try:
        if not isinstance(order, dict):
            return False
        qualities = rules.get("qualities") or QUALITY_COST
        costs = rules.get("costs") or QUALITY_COST
        meals = order["meals"]
        price = float(order["price"])
        if "food" in order and order["food"] not in (rules.get("foods") or ("meat", "veggie")):
            return False
        return (order["spot"] in (rules.get("spots") or SPOTS)
                and order["quality"] in qualities
                and float(rules.get("price_min", 1)) <= price <= float(rules.get("price_max", 30))
                and isinstance(meals, int) and not isinstance(meals, bool) and meals >= 0
                and meals * float(costs.get(order["quality"], QUALITY_COST.get(order["quality"], 0)))
                <= float(rules.get("budget", 500)) + 1e-9)
    except (KeyError, TypeError, ValueError):
        return False


def choose_order(agent, state, log=None):
    """Ask the profit agent for an order; fall back to the heuristic on any failure.

    Returns (order, report). The report is None when the heuristic made the choice; the order is None
    when the agent decided to skip the day."""
    try:
        order, report = agent.plan(state)
        if report and report.get("skip"):
            return None, report  # a day that cannot cover its wages: send nothing (never two in a row)
        if order is None:
            return decide(state), None
        if not valid_order(order, state.get("rules") or {}):
            raise ValueError("invalid agent order {}".format(order))
        return order, report
    except Exception as error:  # The agent must never stop the bot from ordering.
        if log is not None:
            log.write("agent_error", stage="plan", error_type=type(error).__name__,
                      message=str(error)[:200])
        print("Agent plan failed, using fallback: {}".format(error), file=sys.stderr, flush=True)
        return decide(state), None


def observe_state(agent, state, log):
    """Feed the agent a state without letting a learning error interrupt the loop."""
    try:
        agent.observe(state)
    except Exception as error:
        log.write("agent_error", stage="observe", error_type=type(error).__name__,
                  message=str(error)[:200])
        print("Agent observe failed: {}".format(error), file=sys.stderr, flush=True)


def import_snapshot(folder):
    """Start a fresh clone from the team's committed learning instead of from nothing."""
    snapshot = Path(__file__).resolve().parent / "data"
    if (folder / "agent-history.jsonl").exists() or not (snapshot / "agent-history.jsonl").exists():
        return
    folder.mkdir(parents=True, exist_ok=True)
    for name in ("agent-history.jsonl", "agent-model.json", "live.jsonl", "latest-state.json"):
        if (snapshot / name).exists():
            shutil.copyfile(str(snapshot / name), str(folder / name))
    print("Loaded the team's learned data from data/.", flush=True)


def seed_agent(agent, log):
    """Learn from the saved snapshot so a restart does not lose the last completed day."""
    try:
        with (log.path.parent / "latest-state.json").open(encoding="utf-8") as source:
            observe_state(agent, json.load(source), log)
    except (OSError, ValueError):
        pass


def log_decision(log, state, order, report=None):
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
              recent_rivals=len(rivals), estimated_rivals=estimate_rivals(state, spot),
              event_spot=any(event.get("spot") == spot for event in forecast.get("events") or []),
              recent_rival_prices=[truck.get("price") for truck in rivals],
              source="agent" if report else "heuristic",
              expected_profit=(report or {}).get("expected_profit"),
              expected_sold=(report or {}).get("expected_sold"),
              p_loss=(report or {}).get("p_loss"))


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


def estimate_rivals(state, spot):
    """Allow for teams converging on an event or unusually busy spot."""
    previous = (state.get("last_day") or {}).get("spots") or {}
    own_id = (state.get("you") or {}).get("id")
    recent = sum(truck.get("id") != own_id for truck in
                 (previous.get(spot) or {}).get("trucks") or [])
    forecast = state.get("forecast") or {}
    events = forecast.get("events") or []
    market_rivals = max(0, len(state.get("trucks") or []) - 1)
    projected = 0
    if any(event.get("spot") == spot for event in events):
        # A public concert forecast attracted five of six rivals in live play.
        projected = math.ceil(market_rivals * 0.65)
    expected = forecast.get("expected_customers") or {}
    values = [float(value) for value in expected.values() if value is not None]
    if (values and (forecast.get("crowds") or {}).get(spot) == "high"
            and float(expected.get(spot) or 0) >= statistics.median(values) * 1.2):
        # Even without an event, the top forecast drew a premium rival on day 4.
        projected = max(projected, math.ceil(market_rivals * 0.4))
    return max(recent, projected)


def decide(state):
    """Pick a spot with demand and little recent competition."""
    forecast = state.get("forecast") or {}
    expected = forecast.get("expected_customers") or {}
    reputation = (state.get("you") or {}).get("reputation") or {}
    rules = state.get("rules") or {}
    spots = rules.get("spots") or SPOTS

    best_spot = spots[0]
    best_score = float("-inf")
    for spot in spots:
        customers = float(expected.get(spot) or 0)
        # Last day's rivals are only a hint; public events draw teams in.
        rivals = estimate_rivals(state, spot)
        loyal = float(reputation.get(spot, 50))
        score = customers * (0.75 + loyal / 200) / (1 + 0.7 * rivals)
        if score > best_score:
            best_score, best_spot = score, spot

    predicted_rivals = estimate_rivals(state, best_spot)
    # Standard food worked alone; premium rivals won most contested customers.
    quality = "premium" if predicted_rivals and "premium" in (rules.get("qualities") or QUALITY_COST) else "standard"
    target_price = 10.0 if quality == "premium" else 9.5
    price = max(float(rules.get("price_min", 1)), min(target_price, float(rules.get("price_max", 30))))
    # Unsold meals spoil. Avoid committing most of the daily budget to one guess.
    likely_sales = float(expected.get(best_spot) or 60) / (1 + 1.2 * predicted_rivals)
    cost = float((rules.get("costs") or {}).get(quality, QUALITY_COST[quality]))
    budget = float(rules.get("budget", 500))
    meals = min(int(budget // cost), 50, max(12, round(likely_sales * 0.75)))
    return {"spot": best_spot, "price": price, "quality": quality, "meals": meals}


VOTE_TRIES = 3
LATE_SECONDS = 45   # re-plan with the latest planned counts this close to the end of the night


def maybe_vote(agent, state, log, post, voted):
    """Cast the agent's vote once per campaign (or when it changes). Never raises, never retries for long.

    `post` sends {"candidate": id}. Called after the order is in, so a slow or failing vote cannot cost a night."""
    try:
        election = (state.get("politics") or {}).get("election")
        if not election or not (state.get("features") or {}).get("elections") or not hasattr(agent, "vote_decision"):
            return
        decision = agent.vote_decision(state, time_limit=0.3 if state.get("mode") == "final" else 0.6)
        choice = decision.get("choice")
        key = (state.get("mode"), state.get("market"), election.get("id"))
        record = voted.setdefault(key, {"tries": 0, "choice": None, "logged": None})
        if record["logged"] != (choice, decision.get("reason")):
            record["logged"] = (choice, decision.get("reason"))
            log.write("vote_decision", mode=state.get("mode"), market=state.get("market"),
                      election=election.get("id"), choice=choice, reason=decision.get("reason"),
                      current=(state.get("you") or {}).get("vote"), candidates=decision.get("candidates"))
            print("Vote: {} ({})".format(choice or "abstain", decision.get("reason")), flush=True)
        if choice is None or choice == (state.get("you") or {}).get("vote") or record["choice"] == choice \
                or record["tries"] >= VOTE_TRIES:
            return
        record["tries"] += 1
        response = post({"candidate": choice})
        record["choice"] = choice
        log.write("vote_cast", election=election.get("id"), candidate=choice, ok=bool((response or {}).get("ok", True)))
    except urllib.error.HTTPError as error:
        log.write("vote_error", status=error.code, detail=error.read().decode("utf-8", errors="replace")[:200])
        print("Vote failed: HTTP {}".format(error.code), file=sys.stderr, flush=True)
    except Exception as error:  # A vote problem must never touch the orders.
        log.write("agent_error", stage="vote", error_type=type(error).__name__, message=str(error)[:200])
        print("Vote failed: {}".format(error), file=sys.stderr, flush=True)


def seconds_left(state):
    """Seconds until the phase ends by the server's own clock, or None."""
    try:
        parse = lambda text: datetime.datetime.fromisoformat(str(text).replace("Z", "+00:00"))
        return (parse(state["phase_ends_at"]) - parse(state["server_time"])).total_seconds()
    except (KeyError, ValueError, TypeError):
        return None


def planned_key(state):
    """The planned counts tonight (orders in so far per spot) as something comparable."""
    planned = (state.get("forecast") or {}).get("planned")
    return tuple(sorted(planned.items())) if isinstance(planned, dict) else None


def practice(key, days, rivals, seed, log, agent):
    options = {"days": days, "rivals": rivals}
    if seed:
        options["seed"] = seed
    game = request("POST", API + "/practice", key, options, timeout=10)
    base = "https://nightshift.joinbuilders.org" + game["base"]
    voted = {}
    state = request("GET", base + "/state", key)
    while state.get("phase") != "over":
        observe_state(agent, state, log)
        order, report = choose_order(agent, state, log)
        if order is None:
            order = decide(state)  # practice always plays the day
        log_decision(log, state, order, report)
        if report:
            for warning in report.get("warnings") or []:
                log.write("warning", day=state.get("order_for_day"), text=warning)
        # Practice plays the day the moment the order is in, so vote first (live bots vote after ordering).
        maybe_vote(agent, state, log, lambda body: request("POST", base + "/vote", key, body, timeout=3), voted)
        result = request("POST", base + "/orders", key, order, timeout=10)
        state = result.get("state") or request("GET", base + "/state", key)
        log_result(log, state)
        result_line = (state.get("last_day") or {}).get("you") or {}
        print("Day {}: {} | profit ${:.2f} | cash ${:.2f}".format(
            state.get("day"), order, float(result_line.get("profit") or 0),
            float((state.get("you") or {}).get("cash") or 0)), flush=True)
    if state.get("data_quality"):
        # Practice markets may serve deliberately bad data; keep a note of what it was.
        log.write("data_quality", mode=state.get("mode"), market=state.get("market"),
                  data_quality=state.get("data_quality"), rejected=getattr(agent, "rejected", [])[-5:])
        print("Practice data quality: {}".format(state.get("data_quality")), flush=True)
    print("Practice complete. Final cash: ${:.2f}".format(
        float((state.get("you") or {}).get("cash") or 0)), flush=True)


def watched_mtimes():
    stamps = []
    for path in (AGENT_SOURCE, SETTINGS):
        try:
            stamps.append(path.stat().st_mtime)
        except OSError:
            stamps.append(None)
    return tuple(stamps)


def reload_agent(agent, log):
    """Load an edited agent.py into the running bot; keep the old agent if the new code fails."""
    global profit_agent
    try:
        module = importlib.reload(profit_agent)
        fresh = module.Agent(agent.folder, persist=agent.persist, spots=agent.spots,
                             settings_path=agent.settings_path)
    except Exception as error:  # A broken edit must not stop the bot.
        log.write("agent_error", stage="reload", error_type=type(error).__name__, message=str(error)[:300])
        print("Agent reload failed, keeping the running version: {}".format(error), file=sys.stderr, flush=True)
        return agent
    profit_agent = module
    log.write("agent_reloaded")
    print("Agent code reloaded.", flush=True)
    return fresh


def live(key, log, agent):
    version = 0
    seen_results = set()
    planned = {}
    replanned = {}
    voted = {}
    sent = set()
    connected = False
    watched = watched_mtimes()
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
            stamps = watched_mtimes()
            if stamps != watched:
                # New strategy code or team settings: replan tonight's order with them.
                if stamps[0] != watched[0]:
                    agent = reload_agent(agent, log)
                else:
                    print("Agent settings changed; replanning.", flush=True)
                watched = stamps
                planned.clear()
                sent.clear()
            observe_state(agent, state, log)
            if state.get("phase") == "night" and state.get("orders_open"):
                day = state.get("order_for_day")
                existing = (state.get("you") or {}).get("order")
                plan_id = (state.get("mode"), state.get("market"), day)
                # Plan once per day per run, keep the choice if the POST must be retried, and
                # let it replace an older order (for example from before a restart).
                if plan_id not in planned:
                    planned[plan_id] = choose_order(agent, state, log)
                    replanned[plan_id] = planned_key(state)
                    for warning in (planned[plan_id][1] or {}).get("warnings") or []:
                        log.write("warning", day=day, text=warning)
                        print("Warning: {}".format(warning), flush=True)
                left = seconds_left(state)
                if (state.get("mode") == "live" and (state.get("features") or {}).get("planned_counts")
                        and left is not None and 3 < left < LATE_SECONDS and planned_key(state) is not None
                        and replanned.get(plan_id) != planned_key(state)):
                    # Late in the night the planned counts show where rivals really are: plan again.
                    planned[plan_id] = choose_order(agent, state, log)
                    replanned[plan_id] = planned_key(state)
                    sent.discard(plan_id)
                    print("Re-planned with the latest planned counts ({:.0f} s left).".format(left), flush=True)
                order, report = planned[plan_id]
                if order is None and plan_id not in sent:
                    sent.add(plan_id)
                    log.write("order_skipped", mode=state.get("mode"), market=state.get("market"), day=day,
                              expected_profit=(report or {}).get("expected_profit"))
                    print("{} market {} day {}: sitting out (cannot cover wages)".format(
                        state.get("mode"), state.get("market"), day), flush=True)
                elif order is not None and plan_id not in sent and (not existing or any(
                        existing.get(field) != value for field, value in order.items())):
                    log_decision(log, state, order, report)
                    started = time.monotonic()
                    try:
                        response = request("POST", API + "/orders", key, order, timeout=5)
                    except urllib.error.HTTPError as error:
                        detail = error.read().decode("utf-8", errors="replace")
                        if error.code != 403:
                            raise
                        # Wrecked in a riot: the truck is out until its comeback day; do not hammer the API.
                        sent.add(plan_id)
                        log.write("order_refused", status=403, day=day, detail=detail[:200],
                                  wrecked_until_day=(state.get("you") or {}).get("wrecked_until_day"))
                        print("Order refused (403): {}".format(detail[:120]), file=sys.stderr, flush=True)
                        response = None
                    elapsed_ms = round((time.monotonic() - started) * 1000)
                    if response is not None:
                        sent.add(plan_id)
                        log.write("order_submitted", mode=state.get("mode"), market=state.get("market"),
                                  day=day, order=order, latency_ms=elapsed_ms,
                                  source="agent" if report else "heuristic")
                        (state.get("you") or {})["order"] = response.get("order") or order
                        print("{} market {} day {}: submitted {}".format(
                            state.get("mode"), state.get("market"), day, order), flush=True)
                # The order is in (or sat out): only now think about voting, with short timeouts.
                if plan_id in sent or (existing and order is not None):
                    maybe_vote(agent, state, log, lambda body: request("POST", API + "/vote", key, body, timeout=3),
                               voted)
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
        except Exception as error:  # An unexpected state shape must not kill the bot.
            log.write("error", error_type=type(error).__name__, message=str(error)[:200])
            print("Unexpected error: {!r}".format(error), file=sys.stderr, flush=True)
            time.sleep(0.5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--practice", action="store_true", help="run a private practice market")
    parser.add_argument("--days", type=int, default=30, help="practice days")
    parser.add_argument("--rivals", type=int, default=3, help="practice house bots")
    parser.add_argument("--seed", default=None, help="repeatable practice seed")
    parser.add_argument("--log", default=None, help="path to JSONL event log")
    parser.add_argument("--spots", default="all",
                        help="comma-separated spots the agent may choose; 'all' for every spot")
    args = parser.parse_args()
    key = os.environ.get("NIGHTSHIFT_KEY")
    if not key:
        try:
            key = (Path.home() / ".nightshift-key").read_text(encoding="utf-8").strip()
        except OSError:
            key = None
    if not key:
        parser.error("Set NIGHTSHIFT_KEY to your truck key before running the bot")
    log_path = args.log
    if log_path is None:
        folder = Path(__file__).resolve().parent / "logs"
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        log_path = folder / ("practice-{}-{}.jsonl".format(stamp, os.getpid()) if args.practice else "live.jsonl")
    log = EventLog(log_path)
    log.write("run_started", mode="practice" if args.practice else "live", log_path=str(log.path))
    # Practice runs use house bots, so keep them out of the live history.
    spots = None if args.spots.strip().lower() == "all" else [s.strip() for s in args.spots.split(",") if s.strip()]
    if not args.practice:
        import_snapshot(log.path.parent)
    agent = Agent(log.path.parent, persist=not args.practice, spots=spots, settings_path=SETTINGS)
    if args.practice:
        practice(key, args.days, args.rivals, args.seed, log, agent)
    else:
        seed_agent(agent, log)
        live(key, log, agent)


if __name__ == "__main__":
    main()
