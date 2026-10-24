"""Profit agent: learns customers and rivals from every day, then orders for expected profit.

The game publishes how customers choose (value = quality_value * reputation_factor - price,
skip trucks above the price limit or below the minimum quality, buy from the best value).
This module models that directly, fits each spot's price limits to what every truck actually
sold, learns where each rival tends to park and what it charges, and picks the order with the
best risk-adjusted profit. Python 3.8+, standard library only.
"""

import datetime
import json
import math
from pathlib import Path
import random
import time


QUALITIES = ("cheap", "standard", "premium")
DEFAULT_VALUES = {"cheap": 8, "standard": 12, "premium": 16}
DEFAULT_COSTS = {"cheap": 2.5, "standard": 3.75, "premium": 5.5}
REP_STEP = {"cheap": -3.0, "standard": 2.0, "premium": 5.0}
REP_DRIFT = 0.03

# From the published spot guide: typical price limit, spread, and the share of customers
# whose minimum quality is cheap / standard / premium.
SPOT_PRIORS = {
    "highst": {"mu": 12.0, "sd": 3.3, "mix": (0.50, 0.40, 0.10)},
    "oval": {"mu": 12.5, "sd": 2.8, "mix": (0.30, 0.50, 0.20)},
    "union": {"mu": 12.5, "sd": 2.8, "mix": (0.15, 0.65, 0.20)},
    "northres": {"mu": 11.0, "sd": 2.8, "mix": (0.45, 0.47, 0.08)},
    "thompson": {"mu": 12.0, "sd": 2.8, "mix": (0.45, 0.40, 0.15)},
    "stadium": {"mu": 13.0, "sd": 2.8, "mix": (0.25, 0.50, 0.25)},
}
GENERIC_PRIOR = {"mu": 12.0, "sd": 2.8, "mix": (0.35, 0.50, 0.15)}

# Crowd multiplier by actual weather. Outdoor spots lose the most in rain and cold.
WEATHER_EFFECT = {
    "oval": {"clear": 1.1, "cloudy": 1.0, "rain": 0.5, "cold": 0.6},
    "stadium": {"clear": 1.1, "cloudy": 1.0, "rain": 0.55, "cold": 0.6},
    "highst": {"clear": 1.08, "cloudy": 1.0, "rain": 0.65, "cold": 0.75},
    "northres": {"clear": 1.05, "cloudy": 1.0, "rain": 0.75, "cold": 0.85},
    "union": {"clear": 1.0, "cloudy": 1.0, "rain": 0.95, "cold": 0.95},
    "thompson": {"clear": 1.0, "cloudy": 1.0, "rain": 0.95, "cold": 0.95},
}

# Price limits are discretized into bins for a fast fluid model of the customer queue.
BIN = 0.25
LOW = 2.0
HIGH = 24.0
NBINS = int(round((HIGH - LOW) / BIN))

# Calibration grid: shift of each spot's typical limit and its spread.
CALIB_SHIFTS = (-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0)
CALIB_SPREADS = (1.8, 2.3, 2.8, 3.3, 3.8)

LINEUP_SAMPLES = 64
NOISE_POINTS = ((-1.0, 0.3), (0.0, 0.4), (1.0, 0.3))
REP_WEIGHT = 0.35          # share of the future reputation payoff counted today
REP_HORIZON = 6            # days of future sales a reputation gain is credited for
BASE_RISK = 0.3            # weight on expected loss in the objective
RISK_PER_LOSS = 0.25       # extra caution after each consecutive losing day
HOME_WEIGHT = 0.6          # share of a rival's moves that return to spots it has used before
SURPRISE = 0.3             # share of a rival's location spread evenly over every spot


def _event_kind(event):
    text = "{} {}".format(event.get("key") or "", event.get("name") or "").lower()
    for kind in ("gameday", "game day", "midterm", "concert", "late", "rally"):
        if kind in text:
            return kind.replace(" ", "")
    return text.strip()


def customer_prior(spot, events):
    """Spot prior adjusted for the published effects of the day's events."""
    base = SPOT_PRIORS.get(spot, GENERIC_PRIOR)
    mu, sd = base["mu"], base["sd"]
    cheap, standard, premium = base["mix"]
    for event in events or []:
        kind, where = _event_kind(event), event.get("spot")
        if kind == "gameday":
            # Fans pay a little more and lean cheap, most of all at the Stadium.
            mu += 1.0 if spot == where else 0.4
            cheap += 0.12
        elif kind == "midterm" and spot == where:
            premium += 0.25
        elif kind == "concert" and spot == where:
            mu += 0.75
        elif kind == "late" and spot in ("highst", "northres"):
            cheap += 0.2
            mu -= 0.5
    standard = max(0.05, 1.0 - cheap - premium)
    total = cheap + standard + premium
    return mu, sd, (cheap / total, standard / total, premium / total)


_POP_CACHE = {}


def population(mu, sd, mix):
    """Customer mass per (minimum quality, price-limit bin), summing to 1."""
    key = (round(mu, 3), round(sd, 3), tuple(round(m, 4) for m in mix))
    cached = _POP_CACHE.get(key)
    if cached is not None:
        return cached
    cdf = [0.5 * (1 + math.erf((LOW + i * BIN - mu) / (sd * math.sqrt(2)))) for i in range(NBINS + 1)]
    cdf[-1] = 1.0  # Limits above the top bin behave like the top bin.
    shape = [cdf[i + 1] - cdf[i] for i in range(NBINS)]
    total = sum(shape) or 1.0
    result = tuple(tuple(share * mass / total for mass in shape) for share in mix)
    if len(_POP_CACHE) > 4000:
        _POP_CACHE.clear()
    _POP_CACHE[key] = result
    return result


def rep_factor(rep):
    return 0.7 + 0.6 * float(rep) / 100


def _start_bin(price):
    position = (price - LOW) / BIN
    index = int(math.floor(position))
    if index < 0:
        return 0, 1.0
    if index >= NBINS:
        return NBINS, 0.0
    return index, 1.0 - (position - index)


def rank_key(truck):
    # Best value first; when two trucks tie, the cheaper one wins.
    return (-round(truck["value"], 9), truck["price"])


def serve(remaining, truck):
    """Serve one truck from the remaining crowd in place; return meals sold."""
    if truck["value"] < 0 or truck["meals"] <= 0:
        return 0.0
    start, fraction = _start_bin(truck["price"])
    if start >= NBINS:
        return 0.0
    qmax = truck["qi"]
    demand = 0.0
    for q in range(qmax + 1):
        row = remaining[q]
        demand += row[start] * fraction + sum(row[start + 1:])
    if demand <= 1e-9:
        return 0.0
    sold = min(float(truck["meals"]), demand)
    keep = 1.0 - sold / demand
    for q in range(qmax + 1):
        row = remaining[q]
        row[start] *= 1.0 - fraction * (1.0 - keep)
        for b in range(start + 1, NBINS):
            row[b] *= keep
    return sold


def simulate_spot(customers, pop, trucks):
    """Fluid model of one spot: predicted sales per truck (input order) and hungry customers."""
    remaining = [[customers * mass for mass in row] for row in pop]
    order = sorted(range(len(trucks)), key=lambda i: rank_key(trucks[i]))
    sold = [0.0] * len(trucks)
    for i in order:
        sold[i] = serve(remaining, trucks[i])
    return sold, sum(sum(row) for row in remaining)


def _prefix_tables(customers, pop, rivals):
    """Remaining crowd after each prefix of the ranked rivals, with suffix sums for O(1) lookups."""
    remaining = [[customers * mass for mass in row] for row in pop]
    tables = []
    for position in range(len(rivals) + 1):
        cumulative = []
        for q in range(3):
            row = remaining[q]
            suffix = [0.0] * (NBINS + 1)
            running = 0.0
            for b in range(NBINS - 1, -1, -1):
                running += row[b]
                suffix[b] = running
            cumulative.append((list(row), suffix))
        tables.append(cumulative)
        if position < len(rivals):
            serve(remaining, rivals[position])
    return tables


def _lookup(table, price, qi):
    start, fraction = _start_bin(price)
    if start >= NBINS:
        return 0.0
    total = 0.0
    for q in range(qi + 1):
        row, suffix = table[q]
        total += row[start] * fraction + suffix[start + 1]
    return total


class Agent:
    """Keeps day history and learned parameters on disk; plans one order per night."""

    def __init__(self, folder, persist=True):
        self.folder = Path(folder)
        self.persist = persist
        self.history_path = self.folder / "agent-history.jsonl"
        self.model_path = self.folder / "agent-model.json"
        self.plan_path = self.folder / "agent-plan.json"
        self.days = []
        self.seen = set()
        self.model = {"calibration": {}, "ratio": {"n": 0, "sum": 0.0, "sumsq": 0.0},
                      "forecasts": {}, "seen": [], "checks": []}
        self.last_report = None
        self._load()

    # ---------- persistence ----------

    def _load(self):
        try:
            with self.history_path.open(encoding="utf-8") as source:
                for line in source:
                    try:
                        record = json.loads(line)
                    except ValueError:
                        continue
                    key = self._day_key(record)
                    if isinstance(record, dict) and key not in self.seen:
                        self.days.append(record)
                        self.seen.add(key)
        except OSError:
            pass
        try:
            with self.model_path.open(encoding="utf-8") as source:
                stored = json.load(source)
            if isinstance(stored, dict):
                self.model.update(stored)
        except (OSError, ValueError):
            pass
        self.model["seen"] = list(self.model.get("seen") or [])

    def _save_model(self):
        if not self.persist:
            return
        self._write_json(self.model_path, self.model)

    def _write_json(self, path, data):
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + ".tmp")
            temporary.write_text(json.dumps(data, indent=1, default=str) + "\n", encoding="utf-8")
            temporary.replace(path)
        except OSError:
            pass

    @staticmethod
    def _day_key(record):
        return "{}|{}|{}".format(record.get("mode"), record.get("market"), record.get("day"))

    @staticmethod
    def _group(mode):
        # Live and final share the same rival teams; practice uses house bots.
        return "practice" if mode == "practice" else "real"

    # ---------- learning ----------

    def observe(self, state):
        """Record the latest completed day and the coming forecast; learn from anything new."""
        mode, market = state.get("mode"), state.get("market")
        forecast = state.get("forecast") or {}
        if forecast.get("day") is not None and forecast.get("expected_customers"):
            key = "{}|{}|{}".format(mode, market, forecast.get("day"))
            if key not in self.model["forecasts"]:
                self.model["forecasts"][key] = {"expected": forecast.get("expected_customers"),
                                                "weather": forecast.get("weather")}
                if len(self.model["forecasts"]) > 900:
                    for old in sorted(self.model["forecasts"])[:100]:
                        self.model["forecasts"].pop(old, None)
                self._save_model()
        last = state.get("last_day") or {}
        if last.get("day") is None or not last.get("spots"):
            return False
        record = {
            "mode": mode, "market": market, "day": last.get("day"),
            "weather": last.get("weather"), "events": last.get("events") or [],
            "you_id": (state.get("you") or {}).get("id"),
            "spots": {spot: {"customers": data.get("customers"), "hungry": data.get("hungry"),
                             "trucks": [{field: truck.get(field) for field in
                                         ("id", "truck_name", "price", "quality", "meals", "sold")}
                                        for truck in data.get("trucks") or []]}
                      for spot, data in (last.get("spots") or {}).items()},
            "rep_avg": {truck.get("id"): truck.get("reputation_avg") for truck in state.get("trucks") or []},
            "recorded_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        key = self._day_key(record)
        if key in self.seen:
            return False
        self.seen.add(key)
        self.days.append(record)
        if self.persist:
            try:
                self.history_path.parent.mkdir(parents=True, exist_ok=True)
                with self.history_path.open("a", encoding="utf-8") as output:
                    output.write(json.dumps(record, separators=(",", ":"), default=str) + "\n")
            except OSError:
                pass
        self._learn(record, state)
        return True

    def _learn(self, record, state):
        if record["day"] is None:
            return
        key = self._day_key(record)
        if key in self.model["seen"]:
            return
        values = (state.get("rules") or {}).get("values") or DEFAULT_VALUES
        reps = self.reputations(record["mode"], record["market"], before_day=record["day"])
        forecast = self.model["forecasts"].get(key)
        for spot, data in record["spots"].items():
            customers = float(data.get("customers") or 0)
            trucks = data.get("trucks") or []
            if forecast and customers > 0:
                self._learn_ratio(spot, customers, forecast, record.get("weather"))
            if not trucks or customers <= 0:
                continue
            modeled = []
            for truck in trucks:
                quality = truck.get("quality") if truck.get("quality") in QUALITIES else "standard"
                rep = (reps.get(truck.get("id")) or {}).get(spot, 50.0)
                price = float(truck.get("price") or 0)
                modeled.append({"price": price, "qi": QUALITIES.index(quality),
                                "value": float(values.get(quality, DEFAULT_VALUES[quality])) * rep_factor(rep) - price,
                                "meals": float(truck.get("meals") or 0)})
            variance = (2.0 + 0.08 * customers) ** 2
            mu0, sd0, mix = customer_prior(spot, record.get("events"))
            entry = self.model["calibration"].setdefault(spot, {"err": [0.0] * (len(CALIB_SHIFTS) * len(CALIB_SPREADS)), "n": 0})
            index = 0
            for shift in CALIB_SHIFTS:
                for spread in CALIB_SPREADS:
                    predicted, hungry = simulate_spot(customers, population(mu0 + shift, spread, mix), modeled)
                    error = sum((p - float(t.get("sold") or 0)) ** 2 for p, t in zip(predicted, trucks))
                    error += (hungry - float(data.get("hungry") or 0)) ** 2
                    entry["err"][index] += error / variance / 2
                    index += 1
            entry["n"] += 1
        self._check_prediction(record)
        self.model["seen"].append(key)
        self.model["seen"] = self.model["seen"][-2000:]
        self._save_model()

    def _learn_ratio(self, spot, customers, forecast, weather):
        expected = float((forecast.get("expected") or {}).get(spot) or 0)
        if expected <= 0:
            return
        effect = WEATHER_EFFECT.get(spot, WEATHER_EFFECT["union"])
        chances = forecast.get("weather") or {}
        mean_effect = sum(float(chances.get(w, 0)) * effect.get(w, 1.0) for w in effect) or 1.0
        baseline = expected * effect.get(weather, 1.0) / mean_effect
        ratio = math.log(max(1.0, customers) / max(1.0, baseline))
        stats = self.model["ratio"]
        stats["n"] += 1
        stats["sum"] += ratio
        stats["sumsq"] += ratio * ratio

    def _check_prediction(self, record):
        """Compare yesterday's plan to what actually happened, for the dashboard and caution."""
        report = self.last_report
        if not report or report.get("day") != record["day"] or report.get("market") != record["market"]:
            return
        own = None
        for spot, data in record["spots"].items():
            for truck in data.get("trucks") or []:
                if truck.get("id") == record.get("you_id"):
                    own = dict(truck, spot=spot)
        order = report.get("order") or {}
        actual_rivals = {spot: [t.get("truck_name") for t in data.get("trucks") or [] if t.get("id") != record.get("you_id")]
                         for spot, data in record["spots"].items()}
        check = {"mode": record["mode"], "market": record["market"], "day": record["day"],
                 "spot": order.get("spot"), "expected_sold": report.get("expected_sold"),
                 "expected_profit": report.get("expected_profit"),
                 "actual_sold": own.get("sold") if own else None,
                 "actual_profit": None, "expected_rivals": (report.get("rivals") or {}).get(order.get("spot"), {}).get("expected"),
                 "actual_rivals": len(actual_rivals.get(order.get("spot"), []))}
        if own:
            cost = float(DEFAULT_COSTS.get(own.get("quality"), 0)) * float(own.get("meals") or 0)
            check["actual_profit"] = round(float(own.get("sold") or 0) * float(own.get("price") or 0) - cost, 2)
        self.model["checks"] = (self.model.get("checks") or [])[-39:] + [check]

    def spot_params(self, spot, events):
        mu0, sd0, mix = customer_prior(spot, events)
        entry = self.model["calibration"].get(spot)
        best, best_score = (0.0, sd0), float("inf")
        index = 0
        for shift in CALIB_SHIFTS:
            for spread in CALIB_SPREADS:
                penalty = (shift / 1.25) ** 2 / 2 + ((spread - sd0) / 0.75) ** 2 / 2
                score = penalty + (entry["err"][index] if entry else 0.0)
                if score < best_score:
                    best, best_score = (shift, spread), score
                index += 1
        return mu0 + best[0], best[1], mix, (entry or {}).get("n", 0)

    def ratio_stats(self):
        stats = self.model["ratio"]
        n = stats["n"]
        # Shrink toward a slightly low forecast with 15% noise until there is data.
        prior_n, prior_mean, prior_var = 12, -0.05, 0.15 ** 2
        mean = (stats["sum"] + prior_n * prior_mean) / (n + prior_n)
        if n > 1:
            sample_var = max(0.0, stats["sumsq"] / n - (stats["sum"] / n) ** 2)
            var = (sample_var * n + prior_var * prior_n) / (n + prior_n)
        else:
            var = prior_var
        return mean, math.sqrt(var)

    # ---------- rivals ----------

    def market_days(self, mode, market, before_day=None):
        rows = [d for d in self.days if d.get("mode") == mode and d.get("market") == market
                and d.get("day") is not None and (before_day is None or int(d["day"]) < int(before_day))]
        return sorted(rows, key=lambda d: int(d["day"]))

    def reputations(self, mode, market, before_day=None, rep_avg=None):
        """Replay the published reputation rules over every observed day in this market."""
        reps = {}
        last_spot = {}
        previous_day = 0
        for record in self.market_days(mode, market, before_day):
            day = int(record["day"])
            gap = max(0, day - previous_day - 1)
            if gap:
                for table in reps.values():
                    for spot in table:
                        table[spot] += (50.0 - table[spot]) * (1 - (1 - REP_DRIFT) ** gap)
            sold_at = {}
            for spot, data in record["spots"].items():
                for truck in data.get("trucks") or []:
                    if float(truck.get("sold") or 0) >= 1:
                        sold_at[truck.get("id")] = (spot, truck.get("quality"))
            spots = list(record["spots"].keys())
            for truck_id in set(reps) | set(sold_at):
                table = reps.setdefault(truck_id, {spot: 50.0 for spot in spots})
                visited = sold_at.get(truck_id)
                for spot in spots:
                    table.setdefault(spot, 50.0)
                    if visited and spot == visited[0]:
                        table[spot] = min(100.0, max(0.0, table[spot] + REP_STEP.get(visited[1], 0.0)))
                        last_spot[truck_id] = spot
                    else:
                        table[spot] += (50.0 - table[spot]) * REP_DRIFT
            previous_day = day
        for truck_id, average in (rep_avg or {}).items():
            # Days before our history began: put the unexplained reputation where they last sold.
            table = reps.get(truck_id)
            if table is None or average is None or truck_id not in last_spot:
                continue
            gap = 6 * float(average) - sum(table.values())
            spot = last_spot[truck_id]
            table[spot] = min(100.0, max(0.0, table[spot] + gap))
        return reps

    def rival_profiles(self, state):
        """Where each rival is likely to park tomorrow, and what it is likely to sell."""
        mode, market = state.get("mode"), state.get("market")
        group = self._group(mode)
        own_id = (state.get("you") or {}).get("id")
        forecast = state.get("forecast") or {}
        expected = {spot: float(value or 0) for spot, value in (forecast.get("expected_customers") or {}).items()}
        spots = list((state.get("rules") or {}).get("spots") or expected.keys())
        event_spots = [e.get("spot") for e in forecast.get("events") or [] if e.get("spot") in spots]
        top_spot = max(spots, key=lambda s: expected.get(s, 0)) if spots else None
        relevant = [d for d in self.days if self._group(d.get("mode")) == group]
        relevant.sort(key=lambda d: (str(d.get("mode")), int(d.get("market") or 0), int(d.get("day") or 0)))

        stats = {}
        latest_offer = {}
        for index, record in enumerate(relevant):
            where = {}
            for spot, data in record["spots"].items():
                for truck in data.get("trucks") or []:
                    if truck.get("id") != own_id:
                        where[truck.get("id")] = spot
                        latest_offer[truck.get("id")] = dict(truck, spot=spot, market=record.get("market"), mode=record.get("mode"))
            previous = relevant[index - 1] if index else None
            consecutive = (previous is not None and previous.get("mode") == record.get("mode")
                           and previous.get("market") == record.get("market")
                           and int(previous["day"]) == int(record["day"]) - 1)
            before = {}
            if consecutive:
                for spot, data in previous["spots"].items():
                    for truck in data.get("trucks") or []:
                        before[truck.get("id")] = spot
            day_events = [e.get("spot") for e in record.get("events") or []]
            busiest = max(record["spots"], key=lambda s: float(record["spots"][s].get("customers") or 0))
            forecast_record = self.model["forecasts"].get(self._day_key(record))
            if forecast_record and forecast_record.get("expected"):
                busiest = max(forecast_record["expected"], key=lambda s: float(forecast_record["expected"].get(s) or 0))
            for truck_id in set(where) | set(before):
                entry = stats.setdefault(truck_id, {"seen": 0, "active": 0, "stay_n": 0, "stay_k": 0,
                                                    "move_n": 0, "event_k": 0, "event_n": 0, "top_k": 0,
                                                    "spots": {}})
            for truck_id, entry in stats.items():
                entry["seen"] += 1
                if truck_id in where:
                    entry["active"] += 1
                    entry["spots"][where[truck_id]] = entry["spots"].get(where[truck_id], 0) + 1
                if truck_id in where and truck_id in before:
                    entry["stay_n"] += 1
                    if where[truck_id] == before[truck_id]:
                        entry["stay_k"] += 1
                        continue
                if truck_id in where and consecutive:
                    entry["move_n"] += 1
                    if day_events:
                        entry["event_n"] += 1
                        entry["event_k"] += where[truck_id] in day_events
                    entry["top_k"] += where[truck_id] == busiest

        today = self.market_days(mode, market)
        last_record = today[-1] if today else None
        last_spot = {}
        if last_record and int(last_record["day"]) == int(state.get("day") or 0):
            for spot, data in last_record["spots"].items():
                for truck in data.get("trucks") or []:
                    last_spot[truck.get("id")] = spot

        profiles = []
        names = {t.get("id"): t.get("truck_name") for t in state.get("trucks") or []}
        for truck in state.get("trucks") or []:
            truck_id = truck.get("id")
            if truck_id == own_id:
                continue
            # A truck absent from every recorded day has been idle for all of them.
            entry = stats.get(truck_id, {"seen": len(relevant), "active": 0, "stay_n": 0, "stay_k": 0,
                                          "move_n": 0, "event_k": 0, "event_n": 0, "top_k": 0, "spots": {}})
            active = (entry["active"] + 2.0) / (entry["seen"] + 2.5)
            if (entry["seen"] >= 3 and entry["active"] == 0) or "deleted" in str(truck.get("truck_name")).lower():
                active = 0.02  # Never shows up: a deleted or idle team.
            stay = (entry["stay_k"] + 1.0) / (entry["stay_n"] + 2.0)
            # Live day 7: the concert drew only two trucks, not five as on day 3.
            event_rate = (entry["event_k"] + 1.0) / (entry["event_n"] + 3.0)
            top_rate = (entry["top_k"] + 0.6) / (entry["move_n"] + 2.0)
            weights = {spot: max(1.0, expected.get(spot, 0)) ** 1.5 for spot in spots}
            spread = sum(weights.values()) or 1.0
            move = {spot: weights[spot] / spread for spot in spots}
            leftover = 1.0
            mix = {spot: 0.0 for spot in spots}
            if event_spots:
                for spot in event_spots:
                    mix[spot] += event_rate / len(event_spots)
                leftover -= event_rate
            if top_spot:
                share = min(top_rate, max(0.0, leftover))
                mix[top_spot] += share
                leftover -= share
            visits = sum(entry["spots"].values())
            if visits:
                # Teams return to familiar spots (Wagyu Express: High Street on days 2, 5 and 7).
                share = max(0.0, leftover) * HOME_WEIGHT
                for spot, count in entry["spots"].items():
                    if spot in mix:
                        mix[spot] += share * count / visits
                leftover -= share
            for spot in spots:
                mix[spot] += max(0.0, leftover) * move[spot]
            where = {}
            if truck_id in last_spot:
                for spot in spots:
                    where[spot] = (1 - stay) * mix[spot] + (stay if spot == last_spot[truck_id] else 0.0)
            else:
                where = dict(mix)
            # Any active team can surprise us; never treat a spot as certainly empty.
            where = {spot: active * ((1 - SURPRISE) * chance + SURPRISE / len(spots))
                     for spot, chance in where.items()}
            offer = latest_offer.get(truck_id) or {}
            quality = offer.get("quality") if offer.get("quality") in QUALITIES else "premium"
            profiles.append({
                "id": truck_id, "name": names.get(truck_id) or offer.get("truck_name") or truck_id,
                "where": where, "quality": quality,
                "price": float(offer.get("price") or 10.0),
                "meals": float(offer.get("meals") or 90),
                "stay": round(stay, 2), "active": round(active, 2), "known": bool(offer),
            })
        return profiles

    # ---------- planning ----------

    def risk_weight(self, state):
        mode, market = state.get("mode"), state.get("market")
        own_id = (state.get("you") or {}).get("id")
        streak = 0
        for record in reversed(self.market_days(mode, market)):
            profit = None
            for spot, data in record["spots"].items():
                for truck in data.get("trucks") or []:
                    if truck.get("id") == own_id:
                        cost = DEFAULT_COSTS.get(truck.get("quality"), 0) * float(truck.get("meals") or 0)
                        profit = float(truck.get("sold") or 0) * float(truck.get("price") or 0) - cost
            if profit is None:
                continue
            if profit < 0:
                streak += 1
            else:
                break
        return BASE_RISK + RISK_PER_LOSS * min(streak, 4), streak

    def plan(self, state, time_limit=1.0):
        started = time.monotonic()
        rules = state.get("rules") or {}
        spots = list(rules.get("spots") or SPOT_PRIORS.keys())
        values = rules.get("values") or DEFAULT_VALUES
        costs = rules.get("costs") or DEFAULT_COSTS
        qualities = [q for q in (rules.get("qualities") or QUALITIES) if q in QUALITIES]
        budget = float(rules.get("budget", 500))
        price_min, price_max = float(rules.get("price_min", 1)), float(rules.get("price_max", 30))
        forecast = state.get("forecast") or {}
        expected = forecast.get("expected_customers") or {}
        events = forecast.get("events") or []
        chances = {w: float(p) for w, p in (forecast.get("weather") or {"clear": 1.0}).items() if float(p) > 0.03}
        total_chance = sum(chances.values()) or 1.0
        mode, market = state.get("mode"), state.get("market")
        day = int(state.get("order_for_day") or (int(state.get("day") or 0) + 1))
        days_left = max(0, int(state.get("days") or day) - day)
        you = state.get("you") or {}
        own_rep = you.get("reputation") or {}
        rep_avg = {t.get("id"): t.get("reputation_avg") for t in state.get("trucks") or []}
        reps = self.reputations(mode, market, rep_avg=rep_avg)
        profiles = self.rival_profiles(state)
        risk, streak = self.risk_weight(state)
        ratio_mean, ratio_sd = self.ratio_stats()

        seed = (int(market or 0) * 1000 + day) * 7919 + len(profiles)
        rng = random.Random(seed)
        lineups = {spot: {} for spot in spots}
        for _ in range(LINEUP_SAMPLES):
            placed = {spot: [] for spot in spots}
            for profile in profiles:
                roll = rng.random()
                for spot in spots:
                    roll -= profile["where"].get(spot, 0.0)
                    if roll < 0:
                        placed[spot].append(profile["id"])
                        break
            for spot in spots:
                key = tuple(sorted(placed[spot]))
                lineups[spot][key] = lineups[spot].get(key, 0) + 1.0 / LINEUP_SAMPLES
        by_id = {profile["id"]: profile for profile in profiles}

        rivals_report = {}
        per_spot = []
        best = None
        for spot in spots:
            forecast_count = float(expected.get(spot) or 0)
            mu, sd, mix, observations = self.spot_params(spot, events)
            pop = population(mu, sd, mix)
            effect = WEATHER_EFFECT.get(spot, WEATHER_EFFECT["union"])
            mean_effect = sum(chances[w] * effect.get(w, 1.0) for w in chances) / total_chance or 1.0
            crowd_points = {}
            for weather, chance in chances.items():
                for z, weight in NOISE_POINTS:
                    count = forecast_count * effect.get(weather, 1.0) / mean_effect * math.exp(ratio_mean + z * ratio_sd)
                    key = int(round(count))
                    crowd_points[key] = crowd_points.get(key, 0.0) + chance / total_chance * weight

            scenarios = []
            for lineup, lineup_weight in lineups[spot].items():
                trucks = []
                for truck_id in lineup:
                    profile = by_id[truck_id]
                    rep = (reps.get(truck_id) or {}).get(spot, 50.0)
                    quality = profile["quality"]
                    trucks.append({"id": truck_id, "price": profile["price"], "qi": QUALITIES.index(quality),
                                   "value": float(values.get(quality, DEFAULT_VALUES[quality])) * rep_factor(rep) - profile["price"],
                                   "meals": profile["meals"]})
                trucks.sort(key=rank_key)
                for count, crowd_weight in crowd_points.items():
                    scenarios.append((trucks, _prefix_tables(count, pop, trucks), lineup_weight * crowd_weight, count))

            chance_here = {p["name"]: p["where"].get(spot, 0.0) for p in profiles}
            rivals_report[spot] = {
                "expected": round(sum(chance_here.values()), 2),
                "likely": sorted(([name, round(chance, 2)] for name, chance in chance_here.items() if chance >= 0.2),
                                 key=lambda item: -item[1]),
                "customer_limit": round(mu, 2), "limit_spread": round(sd, 2), "observations": observations,
            }

            factor = rep_factor(float(own_rep.get(spot, 50)))
            spot_best = None
            for quality in qualities:
                qi = QUALITIES.index(quality)
                value = float(values.get(quality, DEFAULT_VALUES[quality])) * factor
                cost = float(costs.get(quality, DEFAULT_COSTS[quality]))
                max_meals = int(budget // cost)
                top = min(price_max, value - 0.01)
                prices = set()
                price = max(price_min, math.ceil((cost + 0.25) * 4) / 4)
                while price <= top:
                    prices.add(round(price, 2))
                    price += BIN
                for trucks, _, _, _ in scenarios:
                    for rival in trucks:
                        undercut = round(value - rival["value"] - 0.01, 2)
                        if cost < undercut <= top:
                            prices.add(undercut)
                for price in sorted(prices):
                    our_value = value - price
                    points = []
                    for trucks, tables, weight, _ in scenarios:
                        position = 0
                        for rival in trucks:
                            if rival["value"] > our_value + 1e-9 or (abs(rival["value"] - our_value) <= 1e-9 and rival["price"] <= price):
                                position += 1
                            else:
                                break
                        points.append((_lookup(tables[position], price, qi), weight))
                    candidate = self._best_meals(points, price, cost, max_meals, risk)
                    if candidate is None:
                        continue
                    gain = REP_STEP[quality] * 0.006 * float(values.get(quality, DEFAULT_VALUES[quality]))
                    rep_bonus = REP_WEIGHT * gain * min(days_left, REP_HORIZON) * 45 * candidate["p_sell"]
                    candidate.update(spot=spot, quality=quality, price=round(price, 2),
                                     rep_bonus=round(rep_bonus, 2),
                                     score=candidate["objective"] + rep_bonus)
                    if spot_best is None or candidate["score"] > spot_best["score"]:
                        spot_best = candidate
                if time.monotonic() - started > time_limit:
                    break
            if spot_best:
                spot_best["forecast"] = forecast_count
                per_spot.append(spot_best)
                if best is None or spot_best["score"] > best["score"]:
                    best = spot_best
            if time.monotonic() - started > time_limit:
                break

        if best is None or best["meals"] <= 0:
            return None, None
        order = {"spot": best["spot"], "price": best["price"], "quality": best["quality"], "meals": int(best["meals"])}
        report = {
            "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "mode": mode, "market": market, "day": day, "order": order,
            "expected_sold": round(best["expected_sold"], 1), "expected_profit": round(best["expected_profit"], 2),
            "p_loss": round(best["p_loss"], 3), "expected_loss": round(best["expected_loss"], 2),
            "risk_weight": round(risk, 2), "losing_streak": streak,
            "forecast_bias": round(math.exp(ratio_mean), 3), "forecast_noise": round(ratio_sd, 3),
            "alternatives": [{field: (round(row[field], 2) if isinstance(row[field], float) else row[field])
                              for field in ("spot", "quality", "price", "meals", "expected_sold",
                                            "expected_profit", "p_loss", "rep_bonus", "score", "forecast")}
                             for row in sorted(per_spot, key=lambda r: -r["score"])],
            "rivals": rivals_report,
            "rival_profiles": [{"name": p["name"], "quality": p["quality"], "price": p["price"],
                                "meals": p["meals"], "stay": p["stay"], "active": p["active"],
                                "most_likely": max(p["where"], key=p["where"].get) if p["where"] else None,
                                "chance": round(max(p["where"].values()), 2) if p["where"] else 0}
                               for p in profiles],
            "checks": (self.model.get("checks") or [])[-10:],
            "elapsed_ms": round((time.monotonic() - started) * 1000),
        }
        self.last_report = report
        if self.persist:
            self._write_json(self.plan_path, report)
        return order, report

    @staticmethod
    def _best_meals(points, price, cost, max_meals, risk):
        """Newsvendor choice of meals, cooked a little leaner when recent days lost money."""
        points = [(min(float(demand), float(max_meals)), weight) for demand, weight in points if weight > 0]
        total = sum(weight for _, weight in points)
        if total <= 0 or price <= cost:
            return None
        points.sort()
        shaded = price / (1 + risk * 0.5)
        best_m, best_value = 0, 0.0
        below, below_weight = 0.0, 0.0
        for demand, weight in points:
            meals = int(demand)
            above_weight = total - below_weight
            sold = (below + meals * above_weight) / total
            value = shaded * sold - cost * meals
            if value > best_value:
                best_m, best_value = meals, value
            below += demand * weight
            below_weight += weight
        if best_m <= 0:
            return None
        sold_total = profit_total = loss_total = loss_chance = sell_chance = 0.0
        for demand, weight in points:
            sold = min(best_m, demand)
            profit = price * sold - cost * best_m
            sold_total += sold * weight
            profit_total += profit * weight
            if profit < 0:
                loss_total += -profit * weight
                loss_chance += weight
            if demand >= 1:
                sell_chance += weight
        expected_profit = profit_total / total
        expected_loss = loss_total / total
        return {"meals": best_m, "expected_sold": sold_total / total, "expected_profit": expected_profit,
                "expected_loss": expected_loss, "p_loss": loss_chance / total, "p_sell": sell_chance / total,
                "objective": expected_profit - risk * expected_loss}
