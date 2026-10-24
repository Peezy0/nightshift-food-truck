# Operating the Big Buck bot with Claude Code

This repo runs a live bot for the Builders Night Shift Food Truck Wars challenge. `AGENTS.md` holds the dashboard's design context; `HANDOFF.md` is the human setup guide; game rules are at https://nightshift.joinbuilders.org/kitchen-e551e5/ (the source of truth; they change during the night).

## What runs

- `run.py` (watchdog) keeps exactly one `bot.py` running, restarts it on exit and whenever `bot.py` changes. Output: `logs/bot.out`, `logs/watchdog.out`. Stop everything with `kill $(cat logs/watchdog.pid)`.
- `bot.py` long-polls the game, asks `agent.py` for each night's order (heuristic `decide()` is the fallback), votes after ordering, and saves `logs/latest-state.json`.
- `agent.py` is the profit agent (no AI model). It hot-reloads: an edit to `agent.py` or `agent-settings.json` reaches the live bot within ~25 seconds.
- `dashboard.py` serves a read-only view at http://127.0.0.1:8765.
- The truck key lives in `~/.nightshift-key` or `NIGHTSHIFT_KEY`. Never print, log or commit it. On macOS python.org builds, set `SSL_CERT_FILE=/etc/ssl/cert.pem` for scripts that call the server.

## Rules for Claude

1. **Only one bot per truck.** Before starting `run.py`, check `ps` for `bot.py`/`run.py`. Never run `python3 bot.py` beside the watchdog.
2. **Never experiment in this folder's `agent.py`, `bot.py` or `agent-settings.json`.** The live bot picks them up immediately. Build in a separate worktree (`git worktree add --detach <scratch>/lab HEAD`; this git has no `-q`), using the `strategy-builder` subagent.
3. **Deploy only what passes:** `python3 -m unittest discover -s tests` green in the worktree, a plan on the real `logs/latest-state.json` well under 300 ms, and sensible orders on the regression days in `tests/`. Back up the live files first, copy to a temp name and `mv` into place (atomic), then confirm in `logs/live.jsonl` that the next order has `"source": "agent"` and no `agent_error`. Prefer deploying during a day phase. Commit and push after deploying; the repo must hold the bot that plays.
4. **Ask before outward actions:** POST `/staff`, `/vote` or `/coupons` by hand, deleting logs, or stopping the bot. (Crew is set to 3 employees at $11 on the truck; coupons are never worth it because the final ranks only our own profit.)
5. **Watch for rule changes** whenever the team asks: start a Monitor (30-minute timeout; re-arm on expiry) that every 5 s reads `logs/latest-state.json` and prints when `features` changes or an announcement of kind `rules`/`schedule` appears. On each event, read the docs for that rule, say what it means for the truck, and propose a change; delegate building it to `strategy-builder`.
6. **Investigate losses with data, not guesses:** compare `logs/agent-plan.json` (prediction, alternatives, expected rivals) with `last_day` in `logs/latest-state.json` (what every truck did). Add a losing day as a regression scenario before changing the strategy.
7. **Save credits:** delegate long build-and-test work to `strategy-builder` (Sonnet); keep the main session for review and deploy decisions.
8. **Freeze code at 6 AM.** The final starts 8:30 AM: 20 fresh markets × 30 days, 2 seconds per day, ranked by average profit. Rules live at 8:30 apply to the whole final. Live days don't count.

## Before handing the bot to another computer

Refresh the learned data so the next bot starts informed: copy `logs/agent-history.jsonl`, `logs/agent-model.json`, `logs/live.jsonl` and `logs/latest-state.json` into `data/`, check there is no `fk_` key in them, commit and push. A fresh clone loads `data/` into `logs/` on first start. Stop the old watchdog during a day phase (5 minutes) right after that night's order is in; the new computer starts `python3 run.py` before the next night opens.
