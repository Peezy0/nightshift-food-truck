#!/usr/bin/env python3
"""Keep exactly one live bot running: restart it when it exits or when bot.py changes.

Reads the truck key from NIGHTSHIFT_KEY, or from ~/.nightshift-key when the variable is unset.
Output goes to logs/bot.out. Python 3.8+, standard library only.
"""

import datetime
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parent
KEY_FILE = Path.home() / ".nightshift-key"
WATCHED = (ROOT / "bot.py",)
OUTPUT = ROOT / "logs" / "bot.out"
PID_FILE = ROOT / "logs" / "watchdog.pid"


def say(message):
    print("[watchdog {}] {}".format(datetime.datetime.now().strftime("%H:%M:%S"), message), flush=True)


def load_key():
    key = os.environ.get("NIGHTSHIFT_KEY", "").strip()
    if not key:
        try:
            key = KEY_FILE.read_text(encoding="utf-8").strip()
        except OSError:
            key = ""
    return key


def stamps():
    result = []
    for path in WATCHED:
        try:
            result.append(path.stat().st_mtime)
        except OSError:
            result.append(None)
    return tuple(result)


def start(key):
    env = dict(os.environ, NIGHTSHIFT_KEY=key, PYTHONUNBUFFERED="1")
    if not env.get("SSL_CERT_FILE") and Path("/etc/ssl/cert.pem").exists():
        # python.org builds on macOS ship without CA certificates; use the system bundle.
        env["SSL_CERT_FILE"] = "/etc/ssl/cert.pem"
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    output = OUTPUT.open("a", encoding="utf-8")
    process = subprocess.Popen([sys.executable, str(ROOT / "bot.py")], cwd=str(ROOT), env=env,
                               stdout=output, stderr=subprocess.STDOUT)
    say("bot started (pid {}), output in {}".format(process.pid, OUTPUT))
    return process


def stop(process):
    if process and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()


def main():
    key = load_key()
    if not key:
        sys.exit("Set NIGHTSHIFT_KEY or put the key in {}".format(KEY_FILE))
    PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    PID_FILE.write_text(str(os.getpid()), encoding="utf-8")
    # `kill <watchdog pid>` must stop the bot too, never leave a second bot running.
    signal.signal(signal.SIGTERM, lambda signum, frame: (_ for _ in ()).throw(KeyboardInterrupt()))
    watched = stamps()
    process = start(key)
    failures = 0
    try:
        while True:
            time.sleep(2)
            if stamps() != watched:
                # Let an editor finish writing before restarting on the new code.
                time.sleep(1)
                watched = stamps()
                say("bot.py changed; restarting the bot")
                stop(process)
                process = start(key)
                failures = 0
                continue
            code = process.poll()
            if code is None:
                continue
            failures += 1
            say("bot exited with code {}; restarting".format(code))
            # Back off if it keeps failing fast (for example a bad key), but never give up.
            time.sleep(min(60, 2 * failures))
            process = start(load_key() or key)
    except KeyboardInterrupt:
        say("stopping")
    finally:
        stop(process)
        try:
            PID_FILE.unlink()
        except OSError:
            pass


if __name__ == "__main__":
    main()
