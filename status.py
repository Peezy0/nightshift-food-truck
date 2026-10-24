#!/usr/bin/env python3
"""Explain the latest game state captured by the running bot; no key needed."""

import argparse
import datetime
import json
from pathlib import Path


DEFAULT_PATH = Path(__file__).resolve().parent / "logs" / "latest-state.json"


def money(value):
    return "${:,.2f}".format(float(value or 0))


def line(label, value):
    print("{:20s} {}".format(label + ":", value))


def describe(state):
    print("Game state — captured {}".format(state.get("observed_at") or "unknown time"))
    line("Version", state.get("version"))
    line("Mode", state.get("mode"))
    line("Market", "{} of {}".format(state.get("market"), state.get("markets")))
    line("Day", "{} of {} completed".format(state.get("day"), state.get("days")))
    line("Phase", state.get("phase"))
    line("Server time", state.get("server_time"))
    line("Phase ends", state.get("phase_ends_at"))
    try:
        end = datetime.datetime.fromisoformat(state["phase_ends_at"].replace("Z", "+00:00"))
        server = datetime.datetime.fromisoformat(state["server_time"].replace("Z", "+00:00"))
        line("Time left at capture", "{} seconds".format(max(0, round((end - server).total_seconds()))))
    except (KeyError, TypeError, ValueError):
        pass
    line("Orders open", state.get("orders_open"))
    line("Order for day", state.get("order_for_day"))

    rules = state.get("rules") or {}
    print("\nRules")
    line("Daily food budget", money(rules.get("budget")))
    line("Quality costs", rules.get("costs"))
    line("Quality values", rules.get("values"))
    line("Allowed spots", ", ".join(rules.get("spots") or []))
    line("Price range", "{} to {}".format(money(rules.get("price_min")), money(rules.get("price_max"))))

    forecast = state.get("forecast") or {}
    print("\nForecast for day {}".format(forecast.get("day")))
    line("Weather chances", forecast.get("weather"))
    line("Events", forecast.get("events") or "none")
    for spot, expected in (forecast.get("expected_customers") or {}).items():
        print("  {:9s} {:>5} expected customers | {} crowd".format(
            spot, expected, (forecast.get("crowds") or {}).get(spot, "unknown")))

    you = state.get("you") or {}
    print("\nYour truck")
    line("Name", you.get("truck_name"))
    line("Cash", money(you.get("cash")))
    line("Current order", you.get("order") or "none")
    line("Reputation", you.get("reputation"))

    last = state.get("last_day") or {}
    own = last.get("you") or {}
    print("\nLast completed day: {}".format(last.get("day")))
    line("Actual weather", last.get("weather"))
    line("Actual events", last.get("events") or "none")
    if own.get("spot"):
        line("Your result", "{}: {} of {} meals sold, {} profit".format(
            own.get("spot"), own.get("sold"), own.get("meals"), money(own.get("profit"))))
    else:
        line("Your result", own.get("status") or "no order/result yet")
    print("  Spot results: customers / hungry / trucks (name, price, quality, sold/cooked)")
    for spot, data in (last.get("spots") or {}).items():
        trucks = ["{} {} {} {}/{}".format(truck.get("truck_name"), truck.get("price"),
                                         truck.get("quality"), truck.get("sold"), truck.get("meals"))
                  for truck in data.get("trucks") or []]
        print("  {:9s} {:>4} / {:>4} / {}".format(
            spot, data.get("customers"), data.get("hungry"), "; ".join(trucks) or "none"))

    trucks = state.get("trucks") or []
    print("\nTrucks in market: {}".format(len(trucks)))
    for truck in sorted(trucks, key=lambda item: float(item.get("cash") or 0), reverse=True):
        print("  {:24s} cash {:>10s} | avg reputation {}".format(
            str(truck.get("truck_name") or "unknown")[:24], money(truck.get("cash")),
            truck.get("reputation_avg")))
    announcements = state.get("announcements") or []
    print("\nAnnouncements: {}".format(len(announcements)))
    recent = sorted(announcements, key=lambda item: item.get("at", "") if isinstance(item, dict) else "", reverse=True)
    for announcement in recent[:3]:
        print("  {}".format(announcement.get("text") if isinstance(announcement, dict) else announcement))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", default=str(DEFAULT_PATH))
    args = parser.parse_args()
    path = Path(args.path)
    if not path.exists():
        parser.error("No state snapshot at {}. Pull the update and restart the live bot first.".format(path))
    with path.open(encoding="utf-8") as source:
        describe(json.load(source))


if __name__ == "__main__":
    main()
