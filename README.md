# Big Buck — Food Truck Wars

Bot for the Builders Night Shift Food Truck Wars challenge. Python 3.8 or newer; no third-party packages.

## Start

Keep the truck key private. In your terminal, set `NIGHTSHIFT_KEY` to the key from registration, then run:

```sh
python3 bot.py --practice
python3 bot.py
```

The first command runs a private 30-day practice game. The second stays connected to the live game and the final. Leave it running. If you edit the strategy, restart the bot. Use `--seed same-name` for comparable practice runs.

The bot reads the forecast, estimates competition from the previous day and the coming day's crowd, then chooses a spot. It orders standard meals at $9.50 in a spot expected to be empty, or premium meals at $10 when rivals are expected. It caps production at 50 meals because unsold meals still cost money. The crowding and sales estimates are uncertain: compare later live results and practice runs before changing them further. Each nightly order can be replaced until the night ends.

## Logs and improvement loop

The updated bot writes local JSONL logs in `logs/`. Live events go to `logs/live.jsonl`; each private practice run gets its own file. Logs contain decisions, forecasts, completed results, order response time, and errors. They never contain the truck key, and Git ignores the entire `logs/` directory.

While the live bot is running, open another VS Code terminal in this folder to inspect completed days:

```sh
python3 analyze.py
python3 status.py
```

`analyze.py` summarizes finished days. `status.py` explains the latest state captured by the running bot: version, mode, market and day, phase and server clock, order window, rules, forecast, your truck and current order, the entire last day's spot results, trucks, and announcements. It reads `logs/latest-state.json` locally and needs no truck key in the second terminal. `last_day` is the actual outcome; the forecast is for the *next* day. After the next day starts, run `status.py` again to see how the previous order performed.

## Profit agent

`agent.py` plans each night's order for expected profit. It learns from every completed day: how much customers will pay at each spot, which spots each rival tends to park at and what it charges, and how far the forecast runs high or low. Each night it simulates the customer queue for every spot, quality, price and meal count, and picks the order with the best risk-adjusted profit, cooking a little leaner after losing days. A plan takes about 50 ms, so it fits the final's 2-second limit. If the agent errors or proposes an order that breaks the rules, the bot logs an `agent_error` event and falls back to the original `decide(state)` heuristic. Practice runs use the agent but do not save its history, because house bots are not the real rivals.

It uses no AI model and no credits. It writes three files, all in the git-ignored `logs/` folder:

- `agent-history.jsonl`: one line per completed day (spots, trucks, prices, sales) that it has seen.
- `agent-model.json`: the learned customer price limits, forecast bias, and prediction checks.
- `agent-plan.json`: the latest plan, shown in the dashboard's "Profit agent" section as a forecast, separate from actual results.

A bot that is already running keeps using the old heuristic. To pick up the agent, press `Ctrl+C` in the bot terminal and run `python3 bot.py` again. You can do this at any time, even in the middle of a night: the restarted bot plans once and replaces the existing order for the open day. On startup it also reads `logs/latest-state.json` so the last completed day is not lost.

## Local dashboard

For a browser view, leave `python3 bot.py` running in its first terminal. In a **second** VS Code terminal in this folder, run:

```sh
git pull
python3 dashboard.py
```

Open [http://127.0.0.1:8765](http://127.0.0.1:8765) on the same computer. The dashboard refreshes every five seconds and shows phase timing, your latest order and its food cost, forecast, the full previous day's results, market performance, rules, rivals, and announcements. It reads only local files in `logs/`; it does not need the truck key, send game API requests, or submit orders. The server listens only on your computer. `Ctrl+C` in the dashboard terminal stops the dashboard without stopping the bot.

If it says no state snapshot exists, the running bot is an older version. After its current order is submitted, restart the bot with the latest code in its original terminal. If port 8765 is busy, run `python3 dashboard.py --port 8766` and open that port instead.

For a practice run, pass its file path, for example `python3 analyze.py logs/practice-YYYYMMDDTHHMMSSZ-PID.jsonl`. Look for low sell-through (too much food), frequent stockouts (potential missed sales), low average profit by spot, API errors, and slow order submissions. Use the same `--seed` when comparing two versions of `decide(state)` so weather and events are comparable. Repeat across several seeds and rival counts; a single practice game is not a reliable score against real teams.

To update a running clone, wait until its current nightly order has been submitted, press `Ctrl+C`, run `git pull`, then start `python3 bot.py` again in the same terminal. The key environment variable remains set in that terminal. Keep one live bot running for the truck.

## Important timing

- Night 0 opens at 8:15 PM Eastern; day 1 starts at 8:30 PM.
- Live days repeat every 15 minutes and are for testing, not the final ranking.
- The final starts at 8:30 AM Eastern. The bot must already be running and answer each day in under 2 seconds.

Read the [game rules](https://nightshift.joinbuilders.org/kitchen-e551e5/#/rules) and [API reference](https://nightshift.joinbuilders.org/kitchen-e551e5/#/api).

### Steering the agent without a restart

The agent scores every allowed spot each night by expected profit, chance of a loss, and the reputation it would build, using its predictions of where rivals will park. To add what you know, edit `agent-settings.json` while the bot runs: limit or `avoid` spots, add a dollar bonus to `prefer` a spot, tell it where a rival will be with `rival_hints`, or raise `extra_caution`. The bot notices the change within about 25 seconds and replans that night's order. Edits to `agent.py` also load automatically; if the new code fails to load, the bot keeps the version that was running. Changes to `bot.py` itself still need a restart.

### Watchdog

`python3 run.py` keeps one live bot running: it restarts the bot if it exits and whenever `bot.py` changes, and writes the bot's output to `logs/bot.out`. It reads the key from `NIGHTSHIFT_KEY` or, if that is unset, from `~/.nightshift-key` (outside the repo). Run it instead of `python3 bot.py`, never alongside another bot for the same truck.

### Sharing learned data

`python3 sync_data.py` runs in the background on the computer with the live bot: after each live order it copies the agent's learned files into `data/` and pushes them, so a teammate's fresh clone starts with up-to-date history. It only commits `data/`, skips the final, and refuses to push anything that looks like a key. Run it on one computer at a time.
