# Big Buck — Food Truck Wars

Bot for the Builders Night Shift Food Truck Wars challenge. Python 3.8 or newer; no third-party packages.

## Start

Keep the truck key private. In your terminal, set `NIGHTSHIFT_KEY` to the key from registration, then run:

```sh
python3 bot.py --practice
python3 bot.py
```

The first command runs a private 30-day practice game. The second stays connected to the live game and the final. Leave it running. If you edit the strategy, restart the bot. Use `--seed same-name` for comparable practice runs.

The bot reads the forecast, estimates competition from the previous day, chooses a spot, and orders standard meals at $9.50. It limits production because unsold meals still cost money. Improve `decide(state)` based on practice results and live rivals. Each nightly order can be replaced until the night ends.

## Logs and improvement loop

The updated bot writes local JSONL logs in `logs/`. Live events go to `logs/live.jsonl`; each private practice run gets its own file. Logs contain decisions, forecasts, completed results, order response time, and errors. They never contain the truck key, and Git ignores the entire `logs/` directory.

While the live bot is running, open another VS Code terminal in this folder to inspect completed days:

```sh
python3 analyze.py
```

For a practice run, pass its file path, for example `python3 analyze.py logs/practice-YYYYMMDDTHHMMSSZ-PID.jsonl`. Look for low sell-through (too much food), frequent stockouts (potential missed sales), low average profit by spot, API errors, and slow order submissions. Use the same `--seed` when comparing two versions of `decide(state)` so weather and events are comparable. Repeat across several seeds and rival counts; a single practice game is not a reliable score against real teams.

To update a running clone, wait until its current nightly order has been submitted, press `Ctrl+C`, run `git pull`, then start `python3 bot.py` again in the same terminal. The key environment variable remains set in that terminal. Keep one live bot running for the truck.

## Important timing

- Night 0 opens at 8:15 PM Eastern; day 1 starts at 8:30 PM.
- Live days repeat every 15 minutes and are for testing, not the final ranking.
- The final starts at 8:30 AM Eastern. The bot must already be running and answer each day in under 2 seconds.

Read the [game rules](https://nightshift.joinbuilders.org/kitchen-e551e5/#/rules) and [API reference](https://nightshift.joinbuilders.org/kitchen-e551e5/#/api).
