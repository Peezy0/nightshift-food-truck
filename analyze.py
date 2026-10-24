#!/usr/bin/env python3
"""Summarize a bot JSONL log without contacting the game server."""

import argparse
import json
from collections import defaultdict
from pathlib import Path
import statistics


def summarize(path):
    events = []
    with Path(path).open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                print("Skipping incomplete line {}".format(line_number))

    # A restarted live bot can log the same completed day again.
    results = {}
    for event in events:
        if event.get("event") != "result":
            continue
        key = (event.get("mode"), event.get("market"), event.get("day"))
        if event.get("mode") == "practice":
            key = (event.get("run_id"),) + key
        results[key] = event
    rows = list(results.values())
    if not rows:
        print("No completed days in {} yet.".format(path))
        return

    profit = sum(float(row.get("profit") or 0) for row in rows)
    meals = sum(int(row.get("meals") or 0) for row in rows)
    sold = sum(int(row.get("sold") or 0) for row in rows)
    spoiled = meals - sold
    stockouts = sum(1 for row in rows if (row.get("meals") or 0) > 0
                    and (row.get("sold") or 0) >= row.get("meals"))
    missed = sum(bool(row.get("missed")) for row in rows)
    errors = sum(event.get("event") == "error" for event in events)
    latencies = [event["latency_ms"] for event in events
                 if event.get("event") == "order_submitted" and isinstance(event.get("latency_ms"), (int, float))]

    print("{} | {} completed days".format(path, len(rows)))
    print("Profit ${:,.2f} | average ${:,.2f}/day".format(profit, profit / len(rows)))
    print("Meals: {} sold / {} cooked ({:.1%} sell-through), {} spoiled".format(
        sold, meals, sold / meals if meals else 0, spoiled))
    print("Stockouts: {} | missed days: {} | API errors: {}".format(stockouts, missed, errors))
    if latencies:
        print("Order API latency: median {:.0f} ms, slowest {:.0f} ms".format(
            statistics.median(latencies), max(latencies)))

    by_spot = defaultdict(list)
    for row in rows:
        by_spot[row.get("spot") or "no order"].append(row)
    print("\nBy spot (sorted by average profit):")
    spot_rows = []
    for spot, group in by_spot.items():
        total = sum(float(row.get("profit") or 0) for row in group)
        cooked = sum(int(row.get("meals") or 0) for row in group)
        sales = sum(int(row.get("sold") or 0) for row in group)
        spot_rows.append((total / len(group), spot, len(group), total, sales / cooked if cooked else 0))
    for average, spot, days, total, sell_through in sorted(spot_rows, reverse=True):
        print("  {:9s} {:3d} days | ${:8,.2f}/day | ${:9,.2f} total | {:.0%} sold".format(
            spot, days, average, total, sell_through))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", default=str(Path(__file__).resolve().parent / "logs" / "live.jsonl"))
    args = parser.parse_args()
    if not Path(args.path).exists():
        parser.error("No log at {}. Run the updated bot first.".format(args.path))
    summarize(args.path)


if __name__ == "__main__":
    main()
