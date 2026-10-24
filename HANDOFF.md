# Taking over Big Buck

You are taking over the live bot from a teammate. The bot orders by itself every night and plays the 8:30 AM final. Your job is to keep it running on your computer.

## Before you start

- **Only one bot may run for the truck.** Do not start yours until your teammate says theirs is stopped. Two bots replace each other's orders all night.
- You need Python 3.8 or newer (`python3 --version`; on Windows, `py --version`). No packages to install.
- Your teammate sends you the truck key privately. It never goes in the repo.

## 1. Get the code

```sh
git clone https://github.com/Peezy0/nightshift-food-truck
cd nightshift-food-truck
```

If you already have a copy, run `git pull` in it instead.

## 2. Save the truck key

Mac or Linux:

```sh
printf '%s\n' 'PASTE_THE_KEY_HERE' > ~/.nightshift-key
chmod 600 ~/.nightshift-key
```

Windows (PowerShell):

```powershell
Set-Content -Path "$HOME\.nightshift-key" -Value 'PASTE_THE_KEY_HERE'
```

The file lives in your home folder, outside the repo, so it can never be committed.

## 3. Start the bot (only after your teammate's is stopped)

```sh
python3 run.py
```

On Windows use `py run.py`. Leave this terminal open. You should see `bot started`. Then check its output:

```sh
tail -f logs/bot.out
```

Within a few seconds it prints `Loaded the team's learned data from data/.` and `Connected: live market 1 day N`. On Windows, open `logs\bot.out` in an editor instead of `tail`.

`run.py` is a watchdog: it restarts the bot if it crashes and whenever `bot.py` changes. Strategy changes in `agent.py` and `agent-settings.json` load without any restart.

If you see `CERTIFICATE_VERIFY_FAILED` on a Mac, run `/Applications/Python\ 3.*/Install\ Certificates.command` once, then start `run.py` again.

## 4. Keep the computer awake until the final ends

The bot stops if the computer sleeps, shuts down, or loses internet. Plug it in.

- Mac: in another terminal run `caffeinate -dims` and leave it open.
- Windows: Settings → System → Power → set sleep to Never while plugged in.

The final starts at **8:30 AM Eastern** and the bot must already be running.

## 5. Watch it (optional)

In a second terminal:

```sh
python3 dashboard.py
```

Open http://127.0.0.1:8765. The **Profit agent** panel shows tonight's plan, every spot it compared, which rivals it expects where, and how its past predictions compared with what happened.

## Steering the agent

Edit `agent-settings.json` while the bot runs; it is re-read within about 25 seconds and replans that night's order. No restart.

| Setting | Example | Effect |
|---|---|---|
| `avoid` | `["stadium"]` | Never park there |
| `spots` | `["thompson", "union"]` | Only consider these |
| `prefer` | `{"thompson": 40}` | Treat that spot as worth $40 more |
| `rival_hints` | `{"Wagyu Express": "highst"}` | Tell it where a rival will park |
| `extra_caution` | `1.0` | Weigh possible losses more (fewer meals, safer spots) |

## Stopping

Press `Ctrl+C` in the `run.py` terminal; it stops the bot too. Never leave a stray `python3 bot.py` running alongside it.

## How the profit agent decides

`agent.py` runs inside the bot every night; it calls no AI model and costs nothing to run.

1. **Learns** from every completed day: what each spot's customers pay (fitted to every truck's sales), how far the forecast misses, and each rival's habits (usual spots, whether it chases the day's announced top earner, prices, quality, reputation).
2. **Simulates** about a thousand orders (6 spots × 3 qualities × many prices) against ~64 sampled rival lineups and the weather odds, using the published customer rules. It reads `state.features` every night, so it follows whichever rules are live (fair choice, walk over, salvage).
3. **Orders** the best risk-adjusted option and logs its prediction. After the day, it compares prediction with result.

The dashboard's Profit agent panel shows all of this. If the agent ever fails, `bot.py` falls back to a simple heuristic so a night is never missed.

## How we work on it (with Claude Code)

- **Never edit `agent.py` in the running folder to experiment.** The live bot reloads it within ~25 seconds, so a half-finished edit goes straight into play. Experiment in a separate copy (`git worktree add --detach ../lab HEAD`), and copy it over only once it wins.
- **Prove changes before deploying.** Compare the current and new agent on the same practice seeds (`python3 bot.py --practice --seed S --days 30 --log logs/x.jsonl`, then `python3 analyze.py logs/x.jsonl`). Practice is limited to 10 markets per hour per truck, and practice data can be deliberately bad: check `data_quality` at the end of each market.
- **Use live losses as test cases.** Days 7, 10, 13 and 16 each exposed a flaw (empty-spot bets, chasers undercutting, pricing above the field, three-way splits). New versions must handle them.
- **Steer with `agent-settings.json`,** not code, for quick team knowledge (rival hints, spots to avoid).
- **To save credits,** delegate long build-and-test jobs to a cheaper model (we used Claude Sonnet subagents) and keep the main session for reviewing results and deciding what ships.
- **Read the docs again when rules change.** The organizers change rules mid-night (announced on the TV and in `state.announcements`); the docs at https://nightshift.joinbuilders.org/kitchen-e551e5/ are the source of truth.
- **Freeze at 6 AM.** No code changes after that; the final starts at 8:30 AM. Live days don't count toward the ranking; only the final's average profit per market does.

## Using Claude Code

Open Claude Code in this folder: it reads `CLAUDE.md`, which teaches it how this bot runs, the safety rules (one bot, never experiment on the live files, test before deploying) and how to deploy. It can also launch the `strategy-builder` subagent (`.claude/agents/strategy-builder.md`, runs on Sonnet to save credits) to build and test strategy changes in a separate copy. A good first message: "Read CLAUDE.md, check the bot is running, and start watching for rule changes."

## Watching for rule changes (with Claude Code)

The organizers switch rules on and off during the night. The bot reads `state.features` every night, but a person (or Claude) should still check each change. To get alerted, open Claude Code in this folder and paste:

> Watch for Night Shift rule changes. Start a Monitor (30-minute timeout, re-arm it when it expires) that every 5 seconds reads `logs/latest-state.json` and prints a line whenever `features` changes or a new announcement with kind `rules` or `schedule` appears. When one fires, read the latest docs at https://nightshift.joinbuilders.org/kitchen-e551e5/ for that rule, tell me what it means for our truck, and propose (don't deploy) any change. Never edit `agent.py` in this folder to experiment, since the live bot reloads it; never call /staff, /vote or /coupons without asking me first.

The watch only runs while that Claude Code session stays open; the bot runs either way. Our crew setting (3 employees at $11) and voting (the agent abstains unless a candidate is clearly better for us than the protest risk) are already handled.

## Switching computers

Only one bot may run for the truck. The handover must fit inside one day phase (5 minutes since day 15): the old bot stops right after that night's order is in, and the new one starts before the next night opens. If it misses, the truck sits out one live day, which doesn't count toward the final.

## If something looks wrong

- Dashboard says the snapshot is stale: check the `run.py` terminal and `logs/bot.out`.
- `HTTP 401` in `logs/bot.out`: the key in `~/.nightshift-key` is wrong.
- A losing day is normal now and then; live days don't count toward the final. Repeated losses with the same pattern are worth a look in the dashboard's "Prediction vs actual" table.
