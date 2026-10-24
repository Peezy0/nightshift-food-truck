#!/usr/bin/env python3
"""Local, read-only dashboard for the Night Shift bot. Python 3.8+ only."""

import argparse
import datetime as dt
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import statistics
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parent
ASSETS = {
    "/": (ROOT / "dashboard" / "index.html", "text/html; charset=utf-8"),
    "/style.css": (ROOT / "dashboard" / "style.css", "text/css; charset=utf-8"),
    "/app.js": (ROOT / "dashboard" / "app.js", "text/javascript; charset=utf-8"),
}


def read_json(path):
    try:
        with path.open(encoding="utf-8") as source:
            return json.load(source)
    except (OSError, ValueError):
        return None


def read_events(path):
    events = []
    try:
        with path.open(encoding="utf-8") as source:
            for line in source:
                try:
                    event = json.loads(line)
                except ValueError:
                    continue  # The bot may be in the middle of appending this line.
                if isinstance(event, dict):
                    events.append(event)
    except OSError:
        pass
    return events


def snapshot(log_dir):
    state = read_json(log_dir / "latest-state.json")
    events = read_events(log_dir / "live.jsonl")
    results = {}
    orders = {}
    errors = []
    latencies = []
    for event in events:
        kind = event.get("event")
        key = (event.get("mode"), event.get("market"), event.get("day"))
        if kind == "result" and event.get("day") is not None:
            results[key] = event  # Restarts can report a day twice.
        elif kind == "order_submitted" and event.get("day") is not None:
            orders[key] = event
            if isinstance(event.get("latency_ms"), (int, float)):
                latencies.append(event["latency_ms"])
        elif kind == "error":
            errors.append(event)

    mode = (state or {}).get("mode")
    market = (state or {}).get("market")
    market_rows = [row for key, row in results.items() if key[:2] == (mode, market)]
    market_rows.sort(key=lambda row: int(row.get("day") or 0))
    market_orders = [row for key, row in orders.items() if key[:2] == (mode, market)]
    market_orders.sort(key=lambda row: int(row.get("day") or 0))
    profit = sum(float(row.get("profit") or 0) for row in market_rows)
    cooked = sum(int(row.get("meals") or 0) for row in market_rows)
    sold = sum(int(row.get("sold") or 0) for row in market_rows)
    latest_error = errors[-1] if errors else None
    last_activity = events[-1].get("at") if events else None

    return {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "state": state,
        "plan": read_json(log_dir / "agent-plan.json"),
        "log_exists": (log_dir / "live.jsonl").exists(),
        "activity_at": last_activity,
        "latest_order": market_orders[-1] if market_orders else None,
        "results": market_rows[-40:],
        "summary": {
            "days": len(market_rows),
            "profit": round(profit, 2),
            "average_profit": round(profit / len(market_rows), 2) if market_rows else 0,
            "cooked": cooked,
            "sold": sold,
            "spoiled": max(0, cooked - sold),
            "sell_through": round(sold / cooked, 4) if cooked else None,
            "missed": sum(bool(row.get("missed")) for row in market_rows),
            "stockouts": sum(int(row.get("meals") or 0) > 0 and
                             int(row.get("sold") or 0) >= int(row.get("meals") or 0)
                             for row in market_rows),
            "errors": len(errors),
            "latest_error": latest_error,
            "median_order_ms": round(statistics.median(latencies)) if latencies else None,
        },
    }


class Handler(BaseHTTPRequestHandler):
    log_dir = ROOT / "logs"

    def do_GET(self):
        path = urlsplit(self.path).path
        if path == "/api/snapshot":
            body = json.dumps(snapshot(self.log_dir), separators=(",", ":"), default=str).encode("utf-8")
            content_type = "application/json; charset=utf-8"
        elif path in ASSETS:
            asset, content_type = ASSETS[path]
            try:
                body = asset.read_bytes()
            except OSError:
                self.send_error(500, "Dashboard asset unavailable")
                return
        else:
            self.send_error(404, "Not found")
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; connect-src 'self'; img-src 'self' data:; style-src 'self'; script-src 'self'; object-src 'none'; base-uri 'none'")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format_string, *args):
        pass  # The bot's own terminal already shows order activity.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765, help="local port (default: 8765)")
    args = parser.parse_args()
    try:
        server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    except OSError as error:
        parser.error("Could not start local dashboard: {}. Try --port 8766.".format(error))
    print("Dashboard: http://127.0.0.1:{}/".format(server.server_port), flush=True)
    print("Reading {}. Press Ctrl+C to stop this dashboard.".format(Handler.log_dir), flush=True)
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        print("Dashboard stopped.", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
