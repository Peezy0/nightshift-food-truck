#!/usr/bin/env python3
"""Push the agent's learned data to GitHub after each live order, for a teammate's fresh clone.

Watches logs/live.jsonl for order_submitted events, copies the learned files into data/,
and commits and pushes only data/. Skips the final (orders every few seconds) and never
pushes more than once every few minutes. Python 3.8+, standard library only.
"""

import datetime
import json
from pathlib import Path
import shutil
import subprocess
import time


ROOT = Path(__file__).resolve().parent
LOGS = ROOT / "logs"
DATA = ROOT / "data"
FILES = ("agent-history.jsonl", "agent-model.json", "live.jsonl", "latest-state.json")
MIN_GAP = 240  # seconds between pushes


def say(message):
    print("[sync {}] {}".format(datetime.datetime.now().strftime("%H:%M:%S"), message), flush=True)


def git(*args):
    return subprocess.run(["git"] + list(args), cwd=str(ROOT), capture_output=True, text=True, timeout=60)


def last_order():
    """The newest live order_submitted event, or None."""
    try:
        with (LOGS / "live.jsonl").open("rb") as source:
            source.seek(0, 2)
            source.seek(max(0, source.tell() - 200000))
            lines = source.read().decode("utf-8", errors="replace").splitlines()
    except OSError:
        return None
    for line in reversed(lines):
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if event.get("event") == "order_submitted":
            return event
    return None


def sync(order):
    for name in FILES:
        if (LOGS / name).exists():
            shutil.copyfile(str(LOGS / name), str(DATA / name))
    if any("fk_" in (DATA / name).read_text(encoding="utf-8", errors="replace")
           for name in FILES if (DATA / name).exists()):
        say("refusing to push: a key-like string is in data/")
        return False
    git("add", "data")
    if not git("diff", "--cached", "--quiet", "--", "data").returncode:
        return True  # nothing new
    message = "Refresh learned data after live day {} order".format(order.get("day"))
    result = git("commit", "-m", message, "--", "data")
    if result.returncode:
        say("commit failed: {}".format(result.stderr.strip()[:200]))
        return False
    for attempt in range(2):
        pushed = git("push", "-q")
        if not pushed.returncode:
            say("pushed data for day {}".format(order.get("day")))
            return True
        git("pull", "--rebase", "-q")  # a teammate pushed meanwhile
    say("push failed: {}".format(pushed.stderr.strip()[:200]))
    return False


def main():
    DATA.mkdir(exist_ok=True)
    seen = None
    last_push = 0.0
    while True:
        order = last_order()
        key = order and (order.get("mode"), order.get("market"), order.get("day"))
        if order and key != seen and order.get("mode") == "live" and time.time() - last_push >= MIN_GAP:
            time.sleep(5)  # let the bot finish writing the state for this night
            try:
                if sync(order):
                    seen = key
                    last_push = time.time()
            except Exception as error:  # Never let a sync problem stop the watcher.
                say("sync error: {}".format(error))
        time.sleep(15)


if __name__ == "__main__":
    main()
