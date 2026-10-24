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

TEMPERATURE = 1.5          # fair choice: a truck's pick weight is exp(value / TEMPERATURE)
LEGACY_TEMPERATURE = 0.25  # nearly "best value wins", when the game's fair choice rule is off
WALK_BASE, WALK_SCALE = 0.7, 400.0   # a hungry customer walks with chance 0.7 * exp(-metres / 400)
WALK_ACCEPT = 0.5          # share of walkers at neighbours who would buy from a truck of ours
SALVAGE_RATE = 0.25        # unsold meals refund this share of food cost, when salvage is on
MAX_LINEUPS = 12           # most likely rival lineups kept per spot
LINEUP_SAMPLES = 64
NOISE_POINTS = ((-1.0, 0.3), (0.0, 0.4), (1.0, 0.3))
REP_TRUST = 0.25           # share of the modeled future reputation payoff counted today
REP_DISCOUNT = 0.93        # weight of each further day: rivals and the crowd keep changing
HOME_DAYS = 8              # recent days that show which spot is our home
PHANTOM_PRIOR = (0.25, 4.0)    # prior chance (and weight) of a rival we did not foresee at our spot
PHANTOM_LIMITS = (0.1, 0.5)
BASE_RISK = 0.3            # weight on expected loss in the objective
FINAL_RISK = 0.03          # the final ranks the average of 600 days, so be nearly risk-neutral
RISK_PER_LOSS = 0.25       # extra caution after each consecutive losing day (not in the final)
HOME_WEIGHT = 0.6          # share of a rival's moves that return to spots it has used before
SURPRISE_PRIOR = (0.12, 8.0)   # prior share (and weight) of rival moves nothing predicted
SURPRISE_LIMITS = (0.03, 0.3)
UNDERCUT_PRIOR = (0.4, 0.15)   # prior chance a rival cuts its price when it moves, and by how much

# Walking distances between spots in metres.
DISTANCES = {("oval", "thompson"): 130, ("oval", "union"): 470, ("oval", "stadium"): 590,
             ("oval", "highst"): 440, ("oval", "northres"): 550, ("thompson", "union"): 520,
             ("thompson", "stadium"): 490, ("thompson", "highst"): 580, ("thompson", "northres"): 640,
             ("union", "stadium"): 990, ("union", "highst"): 570, ("union", "northres"): 910,
             ("stadium", "highst"): 980, ("stadium", "northres"): 840, ("highst", "northres"): 410}


def distance(a, b):
    return DISTANCES.get((a, b)) or DISTANCES.get((b, a)) or 600


# Rough customer shares from the spot guide, to size a typical day when tomorrow has an event.
SPOT_SHARE = {"highst": 0.24, "oval": 0.20, "union": 0.18, "northres": 0.16, "thompson": 0.14, "stadium": 0.08}


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


def truck_profit(truck, salvage=False):
    """A truck's day: sales minus food cost, with unsold meals refunded part of it when salvage is on."""
    meals, sold = float(truck.get("meals") or 0), float(truck.get("sold") or 0)
    cost = DEFAULT_COSTS.get(truck.get("quality"), 0.0)
    profit = sold * float(truck.get("price") or 0) - meals * cost
    return profit + (meals - sold) * cost * SALVAGE_RATE if salvage else profit


def features_of(state):
    """Which rule changes are live for the coming day (all three are live now; assumed when unreported)."""
    features = state.get("features") if isinstance(state.get("features"), dict) else {}
    return {"fair_choice": bool(features.get("fair_choice", True)),
            "walk_over": bool(features.get("walk_over", True)),
            "salvage": bool(features.get("salvage", True))}


def _number(value, low=0.0, high=1e9):
    """A usable number: a real number in range, or a money string such as "$9.25"; otherwise None."""
    if isinstance(value, str):
        try:
            value = float(value.strip().lstrip("$"))
        except ValueError:
            return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value != value:
        return None
    return float(value) if low <= value <= high else None


def clean_day(last, spots=None):
    """Repair or drop what cannot be true in a reported day; returns (spots, issues).

    Practice markets can serve corrupted days. Money strings are read as numbers, sales above the
    meals cooked are clipped, and spots, trucks or counts that make no sense are dropped."""
    issues = []
    cleaned = {}
    reported = last.get("spots")
    if not isinstance(reported, dict):
        return cleaned, ["no spots"]
    for spot, data in reported.items():
        if not isinstance(data, dict) or (spots and spot not in spots):
            issues.append("unknown spot {}".format(spot))
            continue
        customers, hungry = _number(data.get("customers"), 0, 3000), _number(data.get("hungry"), 0, 3000)
        walked_in = _number(data.get("walked_in") or 0, 0, 3000)
        walked_out = _number(data.get("walked_out") or 0, 0, 3000)
        if None in (customers, hungry, walked_in, walked_out):
            issues.append("bad customer counts at {}".format(spot))
            continue
        trucks, seen = [], set()
        for truck in data.get("trucks") or []:
            price = _number(truck.get("price"), 0.0, 100.0) if isinstance(truck, dict) else None
            meals = _number(truck.get("meals"), 0.0, 5000.0) if price is not None else None
            sold = _number(truck.get("sold"), 0.0, 5000.0) if meals is not None else None
            if sold is None or truck.get("quality") not in QUALITIES or truck.get("id") in seen:
                issues.append("unusable truck at {}".format(spot))
                continue
            seen.add(truck.get("id"))
            if sold > meals + 1e-6:
                issues.append("sold more than cooked at {}".format(spot))
            trucks.append(dict(truck, price=price, meals=meals, sold=min(sold, meals)))
        if hungry > customers + walked_in + 1e-6:
            issues.append("more hungry than customers at {}".format(spot))
        cleaned[spot] = {"customers": customers, "hungry": min(hungry, customers + walked_in),
                         "walked_in": walked_in, "walked_out": walked_out, "trucks": trucks}
    return cleaned, issues


def clean_forecast(expected, spots):
    """Forecast customers per spot with missing or wildly inflated values replaced by a typical size."""
    valid = {s: _number(v, 0, 5000) for s, v in (expected if isinstance(expected, dict) else {}).items()}
    ratios = sorted(v / SPOT_SHARE[s] for s, v in valid.items() if v and s in SPOT_SHARE)
    scale = ratios[len(ratios) // 2] if ratios else 0.0
    cleaned = {}
    for spot in spots:
        typical = scale * SPOT_SHARE.get(spot, 0.15)
        value = valid.get(spot)
        if value is None:
            value = typical
        elif typical and value > 7 * typical:
            value = typical  # Not even a game day at the Stadium is this big.
        cleaned[spot] = value
    return cleaned


def choice_weight(value, temperature):
    """Share of a customer's pick a truck gets against rivals: exp(value / temperature); 0 if value < 0."""
    return math.exp(min(60.0, value / temperature)) if value >= 0 else 0.0


def crowd_tables(customers, pop):
    """Per minimum quality: customers by price-limit bin and their suffix sums, for O(1) tail lookups."""
    tables = []
    for row in pop:
        scaled = [customers * mass for mass in row]
        suffix = [0.0] * (NBINS + 1)
        running = 0.0
        for b in range(NBINS - 1, -1, -1):
            running += scaled[b]
            suffix[b] = running
        tables.append((scaled, suffix))
    return tables


def _tail(table, price):
    """Customers of one minimum quality whose price limit is at least `price`."""
    start, fraction = _start_bin(price)
    if start >= NBINS:
        return 0.0
    row, suffix = table
    return row[start] * fraction + suffix[start + 1]


def allocate(tables, trucks):
    """Fluid model of one spot under fair choice; returns meals sold per truck and meals left.

    A customer accepts the trucks they can afford at their quality bar and picks one at random in
    proportion to its weight `w`. Customers arrive evenly through the night, so a truck that sells
    out hands its share to the rest of that customer's accepted trucks from then on."""
    n = len(trucks)
    weights = [t["w"] for t in trucks]
    prices = sorted({t["price"] for t, w in zip(trucks, weights) if w > 0})
    sold = [0.0] * n
    left = [float(t["meals"]) for t in trucks]
    if not prices:
        return sold, left
    types = []
    for q in range(3):
        tails = [_tail(tables[q], price) for price in prices] + [0.0]
        for j, price in enumerate(prices):
            mass = tails[j] - tails[j + 1]
            if mass > 1e-9:
                members = [i for i, t in enumerate(trucks)
                           if weights[i] > 0 and t["price"] <= price + 1e-9 and t["qi"] >= q]
                if members:
                    types.append((mass, members))
    remaining = 1.0
    for _ in range(n + 1):
        rate = [0.0] * n
        for mass, members in types:
            live = [i for i in members if left[i] > 1e-9]
            if live:
                total = sum(weights[i] for i in live)
                for i in live:
                    rate[i] += mass * weights[i] / total
        step = remaining
        for i in range(n):
            if rate[i] > 1e-12 and left[i] / rate[i] < step:
                step = left[i] / rate[i]
        if step <= 0 and not any(rate):
            break
        for i in range(n):
            moved = min(left[i], rate[i] * step)
            sold[i] += moved
            left[i] -= moved
        remaining -= step
        if remaining <= 1e-9 or not any(rate):
            break
    return sold, left


def simulate_spot(customers, pop, trucks, temperature=TEMPERATURE):
    """Predicted sales per truck (input order) and hungry customers at one spot."""
    modeled = [dict(t, w=choice_weight(t["value"], temperature)) for t in trucks]
    sold, _ = allocate(crowd_tables(customers, pop), modeled)
    return sold, max(0.0, customers - sum(sold))


class Agent:
    """Keeps day history and learned parameters on disk; plans one order per night."""

    def __init__(self, folder, persist=True, spots=None, settings_path=None):
        self.folder = Path(folder)
        self.persist = persist
        self.spots = tuple(spots or ())
        self.settings_path = Path(settings_path) if settings_path else None
        self.history_path = self.folder / "agent-history.jsonl"
        self.model_path = self.folder / "agent-model.json"
        self.plan_path = self.folder / "agent-plan.json"
        self.days = []
        self.seen = set()
        self.model = {"calibration": {}, "ratio": {"n": 0, "sum": 0.0, "sumsq": 0.0},
                      "forecasts": {}, "seen": [], "checks": []}
        self.last_report = None
        self.surprise = SURPRISE_PRIOR[0]
        self.rejected = []
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
    def _weather(forecast):
        """Weather chances with unusable entries dropped (they are renormalised where used)."""
        weather = forecast.get("weather") if isinstance(forecast.get("weather"), dict) else {}
        chances = {w: _number(p, 0, 1.5) for w, p in weather.items()}
        chances = {w: p for w, p in chances.items() if p is not None and p > 0.03}
        return chances or {"clear": 1.0}

    @staticmethod
    def _group(mode):
        # Live and final share the same rival teams; practice uses house bots.
        return "practice" if mode == "practice" else "real"

    # ---------- learning ----------

    def observe(self, state):
        """Record the latest completed day and the coming forecast; learn from anything new.

        Never raises: a state that cannot be read is skipped."""
        try:
            cleaned = self._clean_state(state)
            last = state.get("last_day") if isinstance(state, dict) else None
            last = last if isinstance(last, dict) else {}
            day = _number(last.get("day"), 0, 1e6)
            cleaned["last_day"] = {"day": None if day is None else int(day),
                                   "weather": last.get("weather") if isinstance(last.get("weather"), str) else None,
                                   "events": [e for e in last.get("events") or [] if isinstance(e, dict)]
                                   if isinstance(last.get("events"), list) else [],
                                   "spots": last.get("spots")}
            return self._observe(cleaned)
        except Exception as error:  # A bad state must never stop the bot.
            self.rejected = (self.rejected + [{"issues": ["unreadable state: {}".format(type(error).__name__)]}])[-20:]
            return False

    def _observe(self, state):
        mode, market = state.get("mode"), state.get("market")
        forecast = state.get("forecast") or {}
        if forecast.get("day") is not None and isinstance(forecast.get("expected_customers"), dict):
            key = "{}|{}|{}".format(mode, market, forecast.get("day"))
            if key not in self.model["forecasts"]:
                known = (state.get("rules") or {}).get("spots") or list(SPOT_PRIORS)
                self.model["forecasts"][key] = {"expected": clean_forecast(forecast.get("expected_customers"), known),
                                                "weather": self._weather(forecast)}
                if len(self.model["forecasts"]) > 900:
                    for old in sorted(self.model["forecasts"])[:100]:
                        self.model["forecasts"].pop(old, None)
                self._save_model()
        last = state.get("last_day") or {}
        if last.get("day") is None or not last.get("spots"):
            return False
        cleaned, issues = clean_day(last, (state.get("rules") or {}).get("spots"))
        if issues:
            self.rejected = (self.rejected + [{"day": last.get("day"), "market": market, "issues": issues[:4]}])[-20:]
        if not cleaned:
            return False
        record = {
            "mode": mode, "market": market, "day": last.get("day"),
            "weather": last.get("weather"), "events": last.get("events") or [],
            "you_id": (state.get("you") or {}).get("id"), "features": features_of(state),
            "spots": {spot: {"customers": data["customers"], "hungry": data["hungry"],
                             "walked_in": data["walked_in"], "walked_out": data["walked_out"],
                             "trucks": [{field: truck.get(field) for field in
                                         ("id", "truck_name", "price", "quality", "meals", "sold")}
                                        for truck in data["trucks"]]}
                      for spot, data in cleaned.items()},
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
        temperature = TEMPERATURE if (record.get("features") or {}).get("fair_choice", True) else LEGACY_TEMPERATURE
        reps = self.reputations(record["mode"], record["market"], before_day=record["day"])
        forecast = self.model["forecasts"].get(key)
        for spot, data in record["spots"].items():
            customers = float(data.get("customers") or 0)
            trucks = data.get("trucks") or []
            if forecast and customers > 0:
                self._learn_ratio(spot, customers, forecast, record.get("weather"))
            if not trucks or customers <= 0:
                continue
            # Walkers from neighbours were hungry customers here too; those who left are still ours to explain.
            customers += float(data.get("walked_in") or 0)
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
                    predicted, hungry = simulate_spot(customers, population(mu0 + shift, spread, mix), modeled,
                                                      temperature)
                    error = sum((p - float(t.get("sold") or 0)) ** 2 for p, t in zip(predicted, trucks))
                    error += (hungry - float(data.get("hungry") or 0) - float(data.get("walked_out") or 0)) ** 2
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
            check["actual_profit"] = round(truck_profit(own, (record.get("features") or {}).get("salvage", True)), 2)
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

    def settings(self):
        """Team overrides from agent-settings.json, re-read every plan; bad files are ignored."""
        if not self.settings_path:
            return {}
        try:
            with self.settings_path.open(encoding="utf-8") as source:
                data = json.load(source)
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

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

    @staticmethod
    def top_earner(record):
        """The spot whose best truck made the most profit that day (what the game announces)."""
        best, best_profit = None, float("-inf")
        salvage = (record.get("features") or {}).get("salvage", True)
        for spot, data in record["spots"].items():
            for truck in data.get("trucks") or []:
                profit = truck_profit(truck, salvage)
                if profit > best_profit:
                    best, best_profit = spot, profit
        return best

    @staticmethod
    def most_hungry(record):
        """The spot where the most customers left without food (live day 15: Oval, then two rivals moved in)."""
        spots = record["spots"]
        return max(spots, key=lambda s: float(spots[s].get("hungry") or 0)) if spots else None

    @staticmethod
    def _blank_stats():
        return {"seen": 0, "active": 0, "stay_n": 0, "stay_k": 0, "move_n": 0, "event_k": 0,
                "event_n": 0, "top_k": 0, "chase_k": 0, "hungry_k": 0, "price_n": 0, "drop_k": 0, "drop_sum": 0.0,
                "spots": {}}

    def rival_profiles(self, state):
        """Where each rival is likely to park tomorrow, and what it is likely to sell."""
        mode, market = state.get("mode"), state.get("market")
        group = self._group(mode)
        own_id = (state.get("you") or {}).get("id")
        forecast = state.get("forecast") or {}
        spots = list((state.get("rules") or {}).get("spots") or SPOT_PRIORS)
        expected = clean_forecast(forecast.get("expected_customers"), spots)
        event_spots = [e.get("spot") for e in forecast.get("events") or [] if e.get("spot") in spots]
        top_spot = max(spots, key=lambda s: expected.get(s, 0)) if spots else None
        relevant = [d for d in self.days if self._group(d.get("mode")) == group]
        relevant.sort(key=lambda d: (str(d.get("mode")), int(d.get("market") or 0), int(d.get("day") or 0)))

        stats = {}
        latest_offer = {}
        unexplained = moves = 0
        for index, record in enumerate(relevant):
            where = {}
            prices_today = {}
            for spot, data in record["spots"].items():
                for truck in data.get("trucks") or []:
                    prices_today[truck.get("id")] = float(truck.get("price") or 0)
                    if truck.get("id") != own_id:
                        where[truck.get("id")] = spot
                        latest_offer[truck.get("id")] = dict(truck, spot=spot, market=record.get("market"), mode=record.get("mode"))
            previous = relevant[index - 1] if index else None
            consecutive = (previous is not None and previous.get("mode") == record.get("mode")
                           and previous.get("market") == record.get("market")
                           and int(previous["day"]) == int(record["day"]) - 1)
            before = {}
            before_price = {}
            yesterday_top = yesterday_hungry = None
            if consecutive:
                yesterday_top = self.top_earner(previous)
                yesterday_hungry = self.most_hungry(previous)
                for spot, data in previous["spots"].items():
                    for truck in data.get("trucks") or []:
                        before[truck.get("id")] = spot
                        before_price[truck.get("id")] = float(truck.get("price") or 0)
            day_events = [e.get("spot") for e in record.get("events") or []]
            busiest = max(record["spots"], key=lambda s: float(record["spots"][s].get("customers") or 0))
            forecast_record = self.model["forecasts"].get(self._day_key(record))
            if forecast_record and forecast_record.get("expected"):
                busiest = max(forecast_record["expected"], key=lambda s: float(forecast_record["expected"].get(s) or 0))
            for truck_id in set(where) | set(before):
                entry = stats.setdefault(truck_id, self._blank_stats())
            for truck_id, entry in stats.items():
                entry["seen"] += 1
                known_spot = where.get(truck_id) in entry["spots"]
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
                    entry["chase_k"] += where[truck_id] == yesterday_top
                    entry["hungry_k"] += where[truck_id] == yesterday_hungry
                    moves += 1
                    unexplained += not (known_spot or where[truck_id] in day_events
                                        or where[truck_id] in (busiest, yesterday_top, yesterday_hungry))
                    if before_price.get(truck_id, 0) > 0 and prices_today.get(truck_id, 0) > 0:
                        change = prices_today[truck_id] / before_price[truck_id] - 1.0
                        entry["price_n"] += 1
                        if change <= -0.05:
                            entry["drop_k"] += 1
                            entry["drop_sum"] += -change

        # Share of moves that no pattern (event, busiest forecast, last top earner, old haunt) explains.
        prior_share, prior_n = SURPRISE_PRIOR
        surprise = min(SURPRISE_LIMITS[1], max(SURPRISE_LIMITS[0],
                       (unexplained + prior_share * prior_n) / (moves + prior_n)))
        self.surprise = surprise

        today = self.market_days(mode, market)
        last_record = today[-1] if today else None
        last_spot = {}
        last_top = last_hungry = None
        if last_record and int(last_record["day"]) == int(state.get("day") or 0):
            last_top = self.top_earner(last_record)
            last_hungry = self.most_hungry(last_record)
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
            entry = stats.get(truck_id) or dict(self._blank_stats(), seen=len(relevant))
            active = (entry["active"] + 2.0) / (entry["seen"] + 2.5)
            if (entry["seen"] >= 3 and entry["active"] == 0) or "deleted" in str(truck.get("truck_name")).lower():
                active = 0.02  # Never shows up: a deleted or idle team.
            stay = (entry["stay_k"] + 1.0) / (entry["stay_n"] + 2.0)
            # Live day 7: the concert drew only two trucks, not five as on day 3.
            event_rate = (entry["event_k"] + 1.0) / (entry["event_n"] + 3.0)
            top_rate = (entry["top_k"] + 0.6) / (entry["move_n"] + 2.0)
            # Live day 10: IBE moved to the spot that earned the most the day before.
            chase_rate = (entry["chase_k"] + 0.75) / (entry["move_n"] + 3.0)
            # Unserved customers draw trucks: they are the best-paying spot nobody covered.
            hungry_rate = (entry["hungry_k"] + 0.75) / (entry["move_n"] + 3.0)
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
            if last_top in mix:
                share = min(chase_rate, max(0.0, leftover))
                mix[last_top] += share
                leftover -= share
            if last_hungry in mix:
                share = min(hungry_rate, max(0.0, leftover))
                mix[last_hungry] += share
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
            where = {spot: active * ((1 - surprise) * chance + surprise / len(spots))
                     for spot, chance in where.items()}
            # Movers tend to undercut (live day 10: IBE cut $9.25-9.75 to $7.25 on arrival).
            cut_chance = (entry["drop_k"] + UNDERCUT_PRIOR[0] * 2) / (entry["price_n"] + 2.0)
            cut_size = (entry["drop_sum"] + UNDERCUT_PRIOR[1] * 2) / (entry["drop_k"] + 2.0)
            offer = latest_offer.get(truck_id) or {}
            quality = offer.get("quality") if offer.get("quality") in QUALITIES else "premium"
            profiles.append({
                "id": truck_id, "name": names.get(truck_id) or offer.get("truck_name") or truck_id,
                "where": where, "last": last_spot.get(truck_id), "quality": quality,
                "price": float(offer.get("price") or 10.0),
                "meals": float(offer.get("meals") or 90),
                "cut_chance": min(0.9, cut_chance), "cut_size": min(0.35, max(0.05, cut_size)),
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
                        profit = truck_profit(truck, (record.get("features") or {}).get("salvage", True))
            if profit is None:
                continue
            if profit < 0:
                streak += 1
            else:
                break
        if mode == "final":
            # The final ranks the average profit of 600 days: expected value is the goal, not safety.
            return FINAL_RISK, streak
        return BASE_RISK + RISK_PER_LOSS * min(streak, 4), streak

    def phantom_odds(self):
        """How often a rival turned up where we parked although the plan expected none (live day 13)."""
        checks = [c for c in self.model.get("checks") or [] if c.get("expected_rivals") is not None]
        surprised = sum(1 for c in checks if float(c["expected_rivals"]) < 0.6 and (c.get("actual_rivals") or 0) >= 1)
        quiet = sum(1 for c in checks if float(c["expected_rivals"]) < 0.6)
        prior, weight = PHANTOM_PRIOR
        return min(PHANTOM_LIMITS[1], max(PHANTOM_LIMITS[0], (surprised + prior * weight) / (quiet + weight)))

    @staticmethod
    def _with_phantom(lineups, chance, by_id):
        """Add an unforeseen low-priced premium truck to each lineup with the given chance."""
        prices = sorted(p["price"] for p in by_id.values() if p["quality"] == "premium" and p["id"] != "phantom")
        by_id["phantom"] = {"id": "phantom", "quality": "premium", "meals": 90.0, "cut_size": 0.0,
                            "price": round(prices[len(prices) // 4], 2) if prices else 8.5}
        result = {}
        for spot, table in lineups.items():
            merged = {}
            for lineup, weight in table.items():
                merged[lineup] = merged.get(lineup, 0.0) + weight * (1.0 - chance)
                extra = tuple(sorted(lineup + (("phantom", False),)))
                merged[extra] = merged.get(extra, 0.0) + weight * chance
            result[spot] = merged
        return result

    def return_odds(self, state, spots):
        """Chance we are at each spot on a later day: our recent visits in this market, with a prior."""
        mode, market = state.get("mode"), state.get("market")
        own_id = (state.get("you") or {}).get("id")
        counts = {spot: 0 for spot in spots}
        recent = self.market_days(mode, market)[-HOME_DAYS:]
        for record in recent:
            for spot, data in record["spots"].items():
                if spot in counts and any(t.get("id") == own_id for t in data.get("trucks") or []):
                    counts[spot] += 1
        return {spot: min(0.9, (counts[spot] + 1.0) / (len(recent) + 3.0)) for spot in spots}

    @staticmethod
    def _clean_state(state):
        """The parts of a state that planning reads, with anything odd or missing replaced by a safe default."""
        state = state if isinstance(state, dict) else {}

        def table(value, keys=None):
            return {k: v for k, v in (value if isinstance(value, dict) else {}).items()
                    if isinstance(k, str) and (keys is None or k in keys)}

        def count(value):
            number = _number(value, 0, 1e6)
            return int(number) if number is not None else None

        raw = table(state.get("rules"))
        spots = [x for x in raw.get("spots") or [] if isinstance(x, str)] if isinstance(raw.get("spots"), list) else []
        rules = {"spots": spots or list(SPOT_PRIORS), "budget": _number(raw.get("budget"), 1, 1e6) or 500,
                 "price_min": _number(raw.get("price_min"), 0, 1000) or 1,
                 "price_max": _number(raw.get("price_max"), 1, 1000) or 30,
                 "qualities": [q for q in raw.get("qualities") or QUALITIES if q in QUALITIES]
                 if isinstance(raw.get("qualities"), list) else list(QUALITIES)}
        for field, defaults in (("values", DEFAULT_VALUES), ("costs", DEFAULT_COSTS)):
            given = table(raw.get(field))
            rules[field] = {q: _number(given.get(q), 0.01, 1000) or defaults[q] for q in QUALITIES}
        rules["qualities"] = rules["qualities"] or list(QUALITIES)
        forecast = table(state.get("forecast"))
        cleaned_forecast = {"day": count(forecast.get("day")),
                            "expected_customers": clean_forecast(forecast.get("expected_customers"), rules["spots"]),
                            "weather": Agent._weather(forecast),
                            "events": [{"key": str(e.get("key") or ""), "name": str(e.get("name") or ""),
                                        "spot": e.get("spot")} for e in forecast.get("events") or []
                                       if isinstance(e, dict) and isinstance(e.get("spot"), str)]
                            if isinstance(forecast.get("events"), list) else []}
        you = table(state.get("you"))
        reputation = {k: _number(v, 0, 100) for k, v in table(you.get("reputation")).items()}
        trucks = [{"id": t.get("id"), "truck_name": str(t.get("truck_name") or ""),
                   "reputation_avg": _number(t.get("reputation_avg"), 0, 100)}
                  for t in state.get("trucks") or [] if isinstance(t, dict) and isinstance(t.get("id"), str)] \
            if isinstance(state.get("trucks"), list) else []
        return {"mode": str(state.get("mode") or ""), "market": count(state.get("market")),
                "day": count(state.get("day")), "days": count(state.get("days")),
                "order_for_day": count(state.get("order_for_day")), "rules": rules,
                "features": features_of(state), "forecast": cleaned_forecast,
                "you": {"id": you.get("id") if isinstance(you.get("id"), str) else None,
                        "reputation": {k: v for k, v in reputation.items() if v is not None}},
                "trucks": trucks}

    def plan(self, state, time_limit=1.0):
        started = time.monotonic()
        state = self._clean_state(state)
        rules = state.get("rules") or {}
        all_spots = list(rules.get("spots") or SPOT_PRIORS.keys())
        settings = self.settings()
        allowed = set(self.spots or all_spots)
        if isinstance(settings.get("spots"), list):
            allowed &= set(settings["spots"])
        allowed -= set(settings.get("avoid") or [])
        # The team can restrict where the truck parks; ignore a restriction that leaves nothing.
        spots = [spot for spot in all_spots if spot in allowed] or all_spots
        prefer = {spot: float(bonus) for spot, bonus in (settings.get("prefer") or {}).items()
                  if isinstance(bonus, (int, float))}
        values = rules.get("values") or DEFAULT_VALUES
        costs = rules.get("costs") or DEFAULT_COSTS
        qualities = [q for q in (rules.get("qualities") or QUALITIES) if q in QUALITIES]
        budget = float(rules.get("budget", 500))
        price_min, price_max = float(rules.get("price_min", 1)), float(rules.get("price_max", 30))
        features = features_of(state)
        temperature = TEMPERATURE if features["fair_choice"] else LEGACY_TEMPERATURE
        salvage = SALVAGE_RATE if features["salvage"] else 0.0
        forecast = state.get("forecast") or {}
        expected = clean_forecast(forecast.get("expected_customers"), all_spots)
        events = [e for e in forecast.get("events") or [] if isinstance(e, dict)]
        chances = self._weather(forecast)
        total_chance = sum(chances.values()) or 1.0
        mode, market = state.get("mode"), state.get("market")
        day = int(state.get("order_for_day") or (int(state.get("day") or 0) + 1))
        days_left = max(0, int(state.get("days") or day) - day)
        you = state.get("you") or {}
        own_rep = you.get("reputation") or {}
        rep_avg = {t.get("id"): t.get("reputation_avg") for t in state.get("trucks") or []}
        reps = self.reputations(mode, market, rep_avg=rep_avg)
        profiles = self.rival_profiles(state)
        hints = {str(name).lower(): spot for name, spot in (settings.get("rival_hints") or {}).items()}
        for profile in profiles:
            hinted = hints.get(str(profile["name"]).lower())
            if hinted in all_spots:
                # A teammate's tip outweighs the learned pattern but still leaves room for error.
                profile["where"] = {spot: 0.85 if spot == hinted else 0.15 / (len(all_spots) - 1)
                                    for spot in all_spots}
        risk, streak = self.risk_weight(state)
        caution = settings.get("extra_caution")
        if isinstance(caution, (int, float)):
            risk = max(0.0, risk + float(caution))
        ratio_mean, ratio_sd = self.ratio_stats()

        seed = (int(market or 0) * 1000 + day) * 7919 + len(profiles)
        # Every spot is modeled, not just the allowed ones: hungry customers walk between neighbours.
        lineups = self._lineups(profiles, all_spots, "where", LINEUP_SAMPLES, random.Random(seed))
        by_id = {profile["id"]: profile for profile in profiles}
        phantom = self.phantom_odds()
        lineups = self._with_phantom(lineups, phantom, by_id)

        model = {}
        for spot in all_spots:
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
            scenarios = self._scenarios(lineups[spot], self._condense(crowd_points), pop, spot, reps, by_id,
                                        values, temperature)
            model[spot] = {"scenarios": scenarios, "mu": mu, "sd": sd, "observations": observations,
                           "forecast": forecast_count}

        walkers = self._walkers(model, lineups, all_spots) if features["walk_over"] else {}

        rivals_report = {}
        per_spot = []
        best = None
        for spot in spots:
            entry = model[spot]
            scenarios, inflow = entry["scenarios"], walkers.get(spot, 0.0)
            chance_here = {p["name"]: p["where"].get(spot, 0.0) for p in profiles}
            rivals_report[spot] = {
                "expected": round(sum(chance_here.values()), 2),
                "likely": sorted(([name, round(chance, 2)] for name, chance in chance_here.items() if chance >= 0.2),
                                 key=lambda item: -item[1]),
                "customer_limit": round(entry["mu"], 2), "limit_spread": round(entry["sd"], 2),
                "observations": entry["observations"], "walk_in": round(inflow, 1),
            }

            factor = rep_factor(float(own_rep.get(spot, 50)))
            spot_best = None
            for quality in qualities:
                qi = QUALITIES.index(quality)
                base_value = float(values.get(quality, DEFAULT_VALUES[quality]))
                value = base_value * factor
                cost = float(costs.get(quality, DEFAULT_COSTS[quality]))
                max_meals = int(budget // cost)
                top = min(price_max, value - 0.01)

                def evaluate(price, scenarios=scenarios, value=value, qi=qi, cost=cost, max_meals=max_meals,
                             inflow=inflow):
                    points = self._points(scenarios, price, value - price, qi, temperature, inflow)
                    return self._best_meals(points, price, cost, max_meals, risk, salvage)

                tried = {}
                step_price = max(price_min, math.ceil((cost + 0.25) * 4) / 4)
                coarse = []
                while step_price <= top:
                    coarse.append(round(step_price, 2))
                    step_price += 0.75
                best_row = None
                for price in coarse:
                    row = evaluate(price)
                    tried[price] = row
                    if row is not None and (best_row is None or row["objective"] > tried[best_row]["objective"]):
                        best_row = price
                if best_row is not None:
                    # Refine around the best coarse price in quarter-dollar steps.
                    for offset in (-0.5, -0.25, 0.25, 0.5):
                        price = round(best_row + offset, 2)
                        if price_min <= price <= top and price > cost and price not in tried:
                            tried[price] = evaluate(price)
                    price = max((p for p in tried if tried[p] is not None), key=lambda p: tried[p]["objective"])
                    candidate = tried[price]
                    rep_bonus = self._reputation_bonus(
                        spot, quality, price, candidate, scenarios, inflow, temperature, base_value, cost,
                        max_meals, risk, salvage, own_rep, days_left, state) * candidate["p_sell"]
                    candidate.update(spot=spot, quality=quality, price=round(price, 2),
                                     rep_bonus=round(rep_bonus, 2),
                                     score=candidate["objective"] + rep_bonus + prefer.get(spot, 0.0))
                    if spot_best is None or candidate["score"] > spot_best["score"]:
                        spot_best = candidate
                if time.monotonic() - started > time_limit:
                    break
            if spot_best:
                spot_best["forecast"] = entry["forecast"]
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
            "mode": mode, "market": market, "day": day, "order": order, "features": features,
            "expected_sold": round(best["expected_sold"], 1), "expected_profit": round(best["expected_profit"], 2),
            "p_loss": round(best["p_loss"], 3), "expected_loss": round(best["expected_loss"], 2),
            "risk_weight": round(risk, 2), "losing_streak": streak,
            "days_left": days_left, "surprise": round(self.surprise, 3), "phantom": round(phantom, 3),
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
            "rejected_days": self.rejected[-5:],
            "settings": {"spots": spots, "prefer": prefer, "rival_hints": settings.get("rival_hints") or {},
                         "extra_caution": caution or 0},
            "elapsed_ms": round((time.monotonic() - started) * 1000),
        }
        self.last_report = report
        if self.persist:
            self._write_json(self.plan_path, report)
        return order, report

    # ---------- scenarios ----------

    @staticmethod
    def _condense(points, size=3):
        """Merge a crowd-size distribution {count: weight} into at most `size` representative points."""
        ordered = sorted(points.items())
        total = sum(weight for _, weight in ordered) or 1.0
        groups = [[0.0, 0.0] for _ in range(size)]
        running = 0.0
        for count, weight in ordered:
            index = min(size - 1, int(size * (running + weight / 2) / total))
            groups[index][0] += count * weight
            groups[index][1] += weight
            running += weight
        return {int(round(c / w)): w / total for c, w in groups if w > 0}

    @staticmethod
    def _lineups(profiles, spots, key, samples, rng):
        """Sample which rivals share each spot, and which of them cut their price on arriving."""
        lineups = {spot: {} for spot in spots}
        for _ in range(samples):
            placed = {spot: [] for spot in spots}
            for profile in profiles:
                roll = rng.random()
                for spot in spots:
                    roll -= profile[key].get(spot, 0.0)
                    if roll < 0:
                        cut = profile["cut_chance"] if profile["last"] and profile["last"] != spot else 0.0
                        placed[spot].append((profile["id"], rng.random() < cut))
                        break
            for spot in spots:
                lineup = tuple(sorted(placed[spot]))
                lineups[spot][lineup] = lineups[spot].get(lineup, 0) + 1.0 / samples
        return lineups

    @staticmethod
    def _scenarios(lineups, crowd_points, pop, spot, reps, by_id, values, temperature):
        """Each likely rival lineup at each crowd size: (rivals, crowd tables, weight)."""
        kept = sorted(lineups.items(), key=lambda item: -item[1])[:MAX_LINEUPS]
        kept_weight = sum(weight for _, weight in kept) or 1.0
        tables = {count: crowd_tables(count, pop) for count in crowd_points}
        scenarios = []
        for lineup, lineup_weight in kept:
            trucks = []
            for truck_id, cut in lineup:
                profile = by_id[truck_id]
                rep = (reps.get(truck_id) or {}).get(spot, 50.0)
                quality = profile["quality"]
                price = round(profile["price"] * (1.0 - profile["cut_size"]), 2) if cut else profile["price"]
                value = float(values.get(quality, DEFAULT_VALUES[quality])) * rep_factor(rep) - price
                trucks.append({"id": truck_id, "price": price, "qi": QUALITIES.index(quality), "value": value,
                               "w": choice_weight(value, temperature), "meals": profile["meals"]})
            for count, crowd_weight in crowd_points.items():
                scenarios.append((trucks, tables[count], lineup_weight / kept_weight * crowd_weight, count))
        return scenarios

    @staticmethod
    def _walkers(model, lineups, spots):
        """Expected hungry customers from other spots who walk to each spot and could buy from us.

        A hungry customer walks to the nearest other spot with a truck they would buy from, with
        chance 0.7 * exp(-metres / 400). Approximate: neighbours' leftovers come from the likely rival
        lineups, and the nearest occupied spot is judged from each spot's chance of any rival."""
        hungry = {}
        occupied = {}
        for spot in spots:
            total = 0.0
            for trucks, tables, weight, count in model[spot]["scenarios"]:
                sold, _ = allocate(tables, trucks)
                total += weight * max(0.0, count - sum(sold))
            hungry[spot] = total
            occupied[spot] = sum(weight for lineup, weight in lineups[spot].items() if lineup)
        inflow = {}
        for spot in spots:
            total = 0.0
            for origin in spots:
                if origin == spot:
                    continue
                reach = WALK_BASE * math.exp(-distance(origin, spot) / WALK_SCALE)
                nearest = 1.0
                for other in spots:
                    if other not in (origin, spot) and distance(origin, other) < distance(origin, spot):
                        nearest *= 1.0 - occupied[other]
                total += hungry[origin] * reach * nearest
            inflow[spot] = total * WALK_ACCEPT
        return inflow

    @staticmethod
    def _points(scenarios, price, our_value, qi, temperature, inflow=0.0):
        """Meals we could sell in each scenario at this price (with walk-ins), with the scenario's weight."""
        if our_value < 0:
            return [(0.0, weight) for _, _, weight, _ in scenarios]
        ours = {"price": price, "qi": qi, "value": our_value, "w": choice_weight(our_value, temperature),
                "meals": 1e9}
        points = []
        for trucks, tables, weight, count in scenarios:
            sold, left = allocate(tables, trucks + [ours])
            demand = sold[-1]
            if inflow > 0:
                # Walkers arrive last and buy only what is left; we share them with rivals still in stock.
                others = sum(t["w"] for t, rest in zip(trucks, left) if rest > 0.5)
                accepting = sum(_tail(tables[q], price) for q in range(qi + 1))
                total = sum(_tail(tables[q], 0.0) for q in range(3)) or 1.0
                demand += inflow * min(1.0, accepting / total) * ours["w"] / (ours["w"] + others)
            points.append((demand, weight))
        return points

    def _reputation_bonus(self, spot, quality, price, candidate, scenarios, inflow, temperature, base_value,
                          cost, max_meals, risk, salvage, own_rep, days_left, state):
        """Worth today of the reputation this order builds at this spot, over the days left.

        A rival cannot profitably undercut a high reputation, so a point of reputation is worth the extra
        profit it would add to this very order, on the later days we are likely to return here (our home).
        It fades to zero as the market ends."""
        if days_left <= 0 or REP_TRUST <= 0:
            return 0.0
        rep = float(own_rep.get(spot, 50))
        gain = min(100.0, max(0.0, rep + REP_STEP[quality])) - (rep + (50.0 - rep) * REP_DRIFT)
        higher = min(100.0, rep + 10.0)
        if gain == 0 or higher <= rep:
            return 0.0
        value = base_value * rep_factor(higher)
        points = self._points(scenarios, price, value - price, QUALITIES.index(quality), temperature, inflow)
        raised = self._best_meals(points, price, cost, max_meals, risk, salvage, fixed_meals=candidate["meals"])
        if raised is None:
            return 0.0
        slope = (raised["objective"] - candidate["objective"]) / (higher - rep)
        stay = self.return_odds(state, [spot])[spot]
        # Reputation keeps its edge while we return (drift fades the rest), over the discounted days left.
        persist = REP_DISCOUNT * (1.0 - REP_DRIFT * (1.0 - stay))
        days = (1.0 - persist ** days_left) / (1.0 - persist)
        return REP_TRUST * stay * slope * gain * days

    @staticmethod
    def _best_meals(points, price, cost, max_meals, risk, salvage=0.0, fixed_meals=None):
        """Newsvendor choice of meals, cooked a little leaner when recent days lost money.

        With salvage, unsold meals refund part of their cost: the same as selling at a lower price
        for a lower effective food cost."""
        points = [(min(float(demand), float(max_meals)), weight) for demand, weight in points if weight > 0]
        total = sum(weight for _, weight in points)
        net_price, net_cost = price - salvage * cost, cost * (1.0 - salvage)
        if total <= 0 or net_price <= net_cost:
            return None
        points.sort()
        shaded = net_price / (1 + risk * 0.5)
        best_m, best_value = 0, 0.0
        below, below_weight = 0.0, 0.0
        for demand, weight in points:
            meals = int(demand)
            above_weight = total - below_weight
            sold = (below + meals * above_weight) / total
            value = shaded * sold - net_cost * meals
            if value > best_value:
                best_m, best_value = meals, value
            below += demand * weight
            below_weight += weight
        if fixed_meals is not None:
            best_m = int(fixed_meals)
        if best_m <= 0:
            return None
        sold_total = profit_total = loss_total = loss_chance = sell_chance = 0.0
        for demand, weight in points:
            sold = min(best_m, demand)
            profit = net_price * sold - net_cost * best_m
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
