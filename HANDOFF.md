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

## If something looks wrong

- Dashboard says the snapshot is stale: check the `run.py` terminal and `logs/bot.out`.
- `HTTP 401` in `logs/bot.out`: the key in `~/.nightshift-key` is wrong.
- A losing day is normal now and then; live days don't count toward the final. Repeated losses with the same pattern are worth a look in the dashboard's "Prediction vs actual" table.
