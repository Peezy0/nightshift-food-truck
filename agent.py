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
    # The eight spots that open later, from the published table (premium share is "wants premium").
    "fisher": {"mu": 14.0, "sd": 2.8, "mix": (0.10, 0.60, 0.30)},
    "kottman": {"mu": 12.5, "sd": 2.8, "mix": (0.25, 0.50, 0.25)},
    "mirror": {"mu": 12.0, "sd": 2.8, "mix": (0.30, 0.50, 0.20)},
    "medcenter": {"mu": 13.0, "sd": 2.8, "mix": (0.25, 0.55, 0.20)},
    "lane": {"mu": 12.0, "sd": 3.3, "mix": (0.50, 0.35, 0.15)},
    "rpac": {"mu": 12.0, "sd": 2.8, "mix": (0.25, 0.50, 0.25)},
    "schott": {"mu": 13.0, "sd": 2.8, "mix": (0.25, 0.55, 0.20)},
    "gateway": {"mu": 11.5, "sd": 2.8, "mix": (0.45, 0.40, 0.15)},
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
    # New spots: Mirror Lake is outdoors, the Medical Center is rain-proof, the rest are guesses.
    "mirror": {"clear": 1.1, "cloudy": 1.0, "rain": 0.5, "cold": 0.6},
    "medcenter": {"clear": 1.0, "cloudy": 1.0, "rain": 1.0, "cold": 1.0},
    "lane": {"clear": 1.05, "cloudy": 1.0, "rain": 0.75, "cold": 0.85},
    "gateway": {"clear": 1.03, "cloudy": 1.0, "rain": 0.85, "cold": 0.9},
    "fisher": {"clear": 1.0, "cloudy": 1.0, "rain": 0.95, "cold": 0.95},
    "kottman": {"clear": 1.0, "cloudy": 1.0, "rain": 0.9, "cold": 0.92},
    "rpac": {"clear": 1.0, "cloudy": 1.0, "rain": 0.95, "cold": 0.95},
    "schott": {"clear": 1.0, "cloudy": 1.0, "rain": 0.9, "cold": 0.92},
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


def make_distance(positions):
    """Metres between spots: from the state's spot_info positions when known (new spots), else the table."""
    def metres(a, b):
        p, q = positions.get(a), positions.get(b)
        if p and q:
            return max(1.0, math.hypot(p[0] - q[0], p[1] - q[1]))
        return distance(a, b)
    return metres


# Rough customer shares from the spot guide, to size a typical day when tomorrow has an event.
SPOT_SHARE = {"highst": 0.24, "oval": 0.20, "union": 0.18, "northres": 0.16, "thompson": 0.14, "stadium": 0.08,
              "fisher": 0.12, "kottman": 0.08, "mirror": 0.10, "medcenter": 0.14, "lane": 0.12, "rpac": 0.09,
              "schott": 0.07, "gateway": 0.11}


def _event_kind(event):
    text = "{} {}".format(event.get("key") or "", event.get("name") or "").lower()
    for kind in ("gameday", "game day", "championship", "midterm", "concert", "late", "rally"):
        if kind in text:
            return "gameday" if kind in ("game day", "championship") else kind
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


# New rules are off until the state says otherwise; the first three are live and assumed when unreported.
FEATURE_DEFAULTS = {"fair_choice": True, "walk_over": True, "salvage": True, "food": False, "moving_costs": False,
                    "new_spots": False, "surprises": False, "football": False, "riots": False, "coupons": False,
                    "staff": False, "planned_counts": False, "elections": False}


def features_of(state):
    """Which rule changes are live for the coming day (the original three assumed on when unreported)."""
    features = state.get("features") if isinstance(state.get("features"), dict) else {}
    return {name: bool(features.get(name, default)) for name, default in FEATURE_DEFAULTS.items()}


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
        free_food = _number(data.get("free_food") or 0, 0, 3000) or 0
        cleaned[spot] = {"customers": customers, "hungry": min(hungry, customers + walked_in),
                         "walked_in": walked_in, "walked_out": walked_out, "trucks": trucks,
                         "distorted": free_food > 0}
    # A giveaway, fire drill or protest takes customers away from one spot: do not learn prices from it.
    for moment in last.get("moments") or []:
        if isinstance(moment, dict) and moment.get("kind") in ("giveaway", "firedrill", "protest") \
                and moment.get("spot") in cleaned:
            cleaned[moment["spot"]]["distorted"] = True
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


def crowd_tables(customers, pop, diets=None):
    """Per minimum quality: customers by price-limit bin and their suffix sums, for O(1) tail lookups.

    With a diet mix {any, meat, veggie} the crowd splits into one such table per diet."""
    if diets:
        return {diet: crowd_tables(customers * share, pop) for diet, share in diets.items() if share > 0}
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


def groups_of(tables):
    """Crowd tables as (diet, per-quality tables) groups; one "any" group unless diets are modeled."""
    return list(tables.items()) if isinstance(tables, dict) else [("any", tables)]


def eats(diet, truck):
    """Whether customers of this diet would buy from the truck: meat-only skip veggie trucks and vice versa."""
    return diet == "any" or (truck.get("food") or "meat") == diet


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
    for diet, group in groups_of(tables):
        for q in range(3):
            tails = [_tail(group[q], price) for price in prices] + [0.0]
            for j, price in enumerate(prices):
                mass = tails[j] - tails[j + 1]
                if mass > 1e-9:
                    members = [i for i, t in enumerate(trucks)
                               if weights[i] > 0 and t["price"] <= price + 1e-9 and t["qi"] >= q
                               and eats(diet, t)]
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


FOODS = ("meat", "veggie")
MEALS_PER_EMPLOYEE = 50    # sales are capped at employees x 50, walk-ins included
DEFAULT_WAGE = 6.0
# Base diet shares (meat-only, veggie-only) from the docs; the forecast scales them by 0.6.
DIET_BASE = {"oval": (0.20, 0.12), "union": (0.20, 0.12), "northres": (0.20, 0.12), "thompson": (0.15, 0.20),
             "stadium": (0.27, 0.03), "highst": (0.24, 0.06), "fisher": (0.40, 0.10), "kottman": (0.10, 0.50),
             "mirror": (0.25, 0.20), "medcenter": (0.25, 0.15), "gateway": (0.25, 0.15), "lane": (0.30, 0.10),
             "rpac": (0.45, 0.10), "schott": (0.35, 0.10)}
GAME_DAY_MEAT_SHIFT = 0.08  # share of every spot's customers who switch from any to meat-only on game days
# Whether forecast.expected_customers already contains the laws in force (checked against practice markets).
FORECAST_HAS_LAWS = True
TERM_DAYS = 8
PROTEST_SHARE = 0.4        # 30-50% of the target spot's customers refuse the protested truck
# Riots: (spot, intensity, radius metres, chance the riot happens) by cause.
RIOTS = {"michigan": ("highst", 0.7, 260.0, 0.45), "title": ("stadium", 0.95, 380.0, 0.8)}

LAW_WORDS = (("veggie_tax", ("veggie",)), ("meat_tax", ("meat",)), ("junk_tax", ("junk",)),
             ("luxury_tax", ("luxury",)), ("price_cap", ("cap",)), ("price_floor", ("floor",)),
             ("curfew", ("curfew",)), ("quiet_core", ("quiet",)), ("tailgate", ("tailgate",)),
             ("crackdown", ("crackdown",)), ("tax_the_top", ("the_top", "top")), ("flat_fee", ("flat",)))
CAMPUS_LAWS = {"curfew": {"highst": 0.75, "northres": 0.75, "oval": 1.1, "thompson": 1.1, "union": 1.1},
               "quiet_core": {"oval": 0.85, "thompson": 0.85, "union": 0.85, "highst": 1.2, "northres": 1.2}}
GAME_LAWS = {"tailgate": {"stadium": 1.3, "highst": 1.3, "schott": 1.3, "rest": 0.9},
             "crackdown": {"stadium": 0.85, "highst": 0.85, "schott": 0.85, "rest": 1.1}}


def law_kind(law):
    """Which of the published laws a law or policy is, from its key (or the start of its text)."""
    key = str(law.get("key") or "").lower() or str(law.get("text") or "").split(":")[0].lower()
    key = key.replace(" ", "_").replace("-", "_")
    for kind, words in LAW_WORDS:
        if any(word in key for word in words):
            return kind
    return None


def _walk_numbers(value, depth=0):
    """Every (key, number) pair nested in a params structure."""
    if isinstance(value, dict) and depth < 4:
        for key, inner in value.items():
            number = _number(inner, -1e6, 1e6)
            if number is not None:
                yield str(key).lower(), number
            else:
                for pair in _walk_numbers(inner, depth + 1):
                    yield pair


def _param(params, words, default, low=0.0, high=1e6):
    for key, number in _walk_numbers(params):
        if any(word in key for word in words) and low <= number <= high:
            return number
    return default


def _spot_table(params, defaults):
    """A {spot: multiplier} table from a law's params when it carries one, else the published numbers."""
    found = {k: v for k, v in _walk_numbers(params) if k in SPOT_SHARE or k in ("rest", "other", "others")}
    if sum(1 for k in found if k in SPOT_SHARE) >= 2:
        found.setdefault("rest", found.pop("other", found.pop("others", 1.0)))
        return found
    return defaults


def law_policy(laws, game_day=False):
    """What the laws in force do to a truck: price cap/floor, per-meal taxes, crowd multipliers, money terms.

    `laws` are the politics.laws (or a candidate's policies): {key, text, params}. Published numbers are
    the fallback when a law's params are not in the shape expected here."""
    policy = {"cap": None, "floor": None, "tax_quality": {}, "tax_food": {}, "tables": [],
              "top": None, "fixed": 0.0, "riot": 1.0, "kinds": []}
    for law in laws if isinstance(laws, list) else []:
        if not isinstance(law, dict):
            continue
        kind = law_kind(law)
        params = law.get("params") if isinstance(law.get("params"), dict) else {}
        if kind is None:
            continue
        policy["kinds"].append(kind)
        if kind == "price_cap":
            policy["cap"] = min(policy["cap"] or 1e9, _param(params, ("max", "cap", "price"), 12.0, 1, 100))
        elif kind == "price_floor":
            policy["floor"] = max(policy["floor"] or 0.0, _param(params, ("min", "floor", "price"), 7.5, 1, 100))
        elif kind in ("meat_tax", "veggie_tax", "junk_tax", "luxury_tax"):
            defaults = {"meat_tax": {"meat": 0.5, "veggie": -0.5}, "veggie_tax": {"veggie": 0.5, "meat": -0.5},
                        "junk_tax": {"cheap": 0.5, "premium": -0.5}, "luxury_tax": {"premium": 0.5, "cheap": -0.5}}[kind]
            given = params.get("per_meal") if isinstance(params.get("per_meal"), dict) else {}
            table = {k: _number(v, -50, 50) for k, v in given.items()}
            table = {k: v for k, v in table.items() if v is not None and k in QUALITIES + FOODS} or defaults
            for name, amount in table.items():
                bucket = policy["tax_quality"] if name in QUALITIES else policy["tax_food"]
                bucket[name] = bucket.get(name, 0.0) + amount
        elif kind in CAMPUS_LAWS or (kind in GAME_LAWS and game_day):
            policy["tables"].append(_spot_table(params, (CAMPUS_LAWS.get(kind) or GAME_LAWS.get(kind))))
            if kind == "tailgate":
                policy["riot"] *= 1.5
        elif kind == "crackdown":
            policy["riot"] = 0.0
        elif kind in ("tax_the_top", "flat_fee"):
            threshold = _param(params, ("above", "threshold", "over"), 500.0, 1, 1e5)
            rate = _param(params, ("rate", "share", "pct", "percent"), 0.10, 0.0, 100.0)
            rate = rate / 100 if rate > 1 else rate
            fixed = _param(params, ("fee", "per_truck", "subsidy", "flat", "amount"), 20.0, 0.0, 500.0)
            policy["top"] = (threshold, rate if kind == "tax_the_top" else -rate)
            policy["fixed"] += -fixed if kind == "tax_the_top" else fixed
    return policy


def crowd_mult(policy, spot):
    """How the laws in force scale the crowd at a spot."""
    mult = 1.0
    for table in policy["tables"]:
        mult *= table.get(spot, table.get("rest", 1.0))
    return mult


def charged_price(price, policy):
    """The price customers pay: a cap lowers it, a floor raises it."""
    if policy.get("cap") is not None:
        price = min(price, policy["cap"])
    if policy.get("floor") is not None:
        price = max(price, policy["floor"])
    return price


def meal_tax(policy, quality, food):
    """Tax per meal sold under the laws (negative: rebate)."""
    return policy["tax_quality"].get(quality, 0.0) + (policy["tax_food"].get(food, 0.0) if food else 0.0)


def riot_warnings(forecast, positions):
    """Riot risks the forecast warns about: [{spot, x, z, intensity, radius, chance}].

    The warning's shape is not published, so look for the word riot in events and warning-like lists, take
    explicit numbers when present, and otherwise infer the cause (title -> Stadium, else Michigan -> High Street)."""
    found = []
    items = []
    for field in ("events", "warnings", "riots", "riot", "alerts"):
        value = forecast.get(field)
        items += [(field, v) for v in (value if isinstance(value, list) else [value]) if v]
    for field, item in items:
        text = json.dumps(item, default=str).lower() if not isinstance(item, str) else item.lower()
        if "riot" not in text and field not in ("riots", "riot"):
            continue
        item = item if isinstance(item, dict) else {}
        title = any(word in text for word in ("title", "championship", "national")) \
            and "high street" not in text and "highst" not in text and "michigan" not in text
        spot, intensity, radius, chance = RIOTS["title" if title else "michigan"]
        center = item.get("center") if isinstance(item.get("center"), dict) else {}
        # A game's own spot is the Stadium; only an item about the riot itself says where it will be.
        if (field in ("riots", "riot") or "riot" in str(item.get("key") or "").lower()) \
                and isinstance(item.get("spot"), str) and item["spot"] in positions:
            spot = item["spot"]
        x, z = center.get("x"), center.get("z")
        if _number(x, -1e5, 1e5) is None or _number(z, -1e5, 1e5) is None:
            x, z = (positions.get(spot) or (0.0, 0.0))
        found.append({"spot": spot, "x": float(x), "z": float(z),
                      "intensity": _number(item.get("intensity"), 0.01, 1.0) or intensity,
                      "radius": _number(item.get("radius"), 50, 5000) or radius,
                      "chance": _number(item.get("chance") or item.get("probability"), 0.01, 1.0) or chance})
    return found


def riot_cost(warnings, position, day_profit, policy):
    """Expected cost of parking at `position` when a riot may follow: wreck (out 2 days) or a repair bill."""
    total = 0.0
    for risk in warnings:
        if position is None:
            continue
        d = math.hypot(position[0] - risk["x"], position[1] - risk["z"])
        hit = min(1.0, risk["intensity"] * policy.get("riot", 1.0) * math.exp(-d / risk["radius"]))
        repair = 30.0 + 120.0 * math.exp(-d / risk["radius"])
        total += risk["chance"] * hit * (0.3 * 2.0 * max(0.0, day_profit) + 0.7 * repair)
    return total


def diet_mix(raw, spot, game_day):
    """{any, meat, veggie} shares for a spot from the forecast; game days shift 8% of customers to meat-only."""
    mix = raw.get(spot) if isinstance(raw, dict) else None
    if not isinstance(mix, dict):
        return None
    meat, veggie = _number(mix.get("meat"), 0, 1), _number(mix.get("veggie"), 0, 1)
    if meat is None or veggie is None or meat + veggie > 1:
        return None
    if game_day:
        # The forecast may already contain the shift: only add it when meat looks like the unshifted base.
        base = DIET_BASE.get(spot)
        shifted = base is not None and meat >= 0.6 * base[0] + GAME_DAY_MEAT_SHIFT * 0.6
        if not shifted:
            meat += min(GAME_DAY_MEAT_SHIFT, 1.0 - meat - veggie)
    return {"any": max(0.0, 1.0 - meat - veggie), "meat": meat, "veggie": veggie}


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
        self._vote_cache = {}
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
                                   "spots": last.get("spots"),
                                   "moments": [m for m in last.get("moments") or [] if isinstance(m, dict)]
                                   if isinstance(last.get("moments"), list) else []}
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
                             "distorted": data["distorted"],
                             "trucks": [{field: truck.get(field) for field in
                                         ("id", "truck_name", "price", "quality", "meals", "sold", "food")}
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
            if data.get("distorted"):
                continue
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
                "id": truck_id, "name": str(names.get(truck_id) or offer.get("truck_name") or truck_id),
                "where": where, "last": last_spot.get(truck_id), "quality": quality,
                "price": float(offer.get("price") or 10.0),
                "meals": float(offer.get("meals") or 90),
                "food": offer.get("food") if offer.get("food") in FOODS else "meat",
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
    def _clean_politics(raw):
        """Laws in force, the office holder and the campaign, with anything unreadable dropped."""
        raw = raw if isinstance(raw, dict) else {}

        def policies(items):
            return [{"key": str(p.get("key") or ""), "text": str(p.get("text") or "")[:200],
                     "params": p.get("params") if isinstance(p.get("params"), dict) else {}}
                    for p in items or [] if isinstance(p, dict)] if isinstance(items, list) else []

        office = raw.get("office") if isinstance(raw.get("office"), dict) else None
        holder = (office or {}).get("holder") if isinstance((office or {}).get("holder"), dict) else {}
        election = raw.get("election") if isinstance(raw.get("election"), dict) else None
        result = {"laws": policies(raw.get("laws")), "office": None, "election": None}
        if office:
            result["office"] = {"holder": str(holder.get("name") or ""), "holder_id": holder.get("id"),
                                "since_day": office.get("since_day"), "until_day": office.get("until_day"),
                                "crisis": bool(office.get("crisis")), "policies": policies(holder.get("policies"))}
        if election:
            candidates = [{"id": str(c.get("id")), "name": str(c.get("name") or ""),
                           "policies": policies(c.get("policies"))}
                          for c in election.get("candidates") or [] if isinstance(c, dict) and c.get("id") is not None] \
                if isinstance(election.get("candidates"), list) else []
            tally = election.get("tally") if isinstance(election.get("tally"), dict) else {}
            votes = election.get("votes") if isinstance(election.get("votes"), dict) else {}
            result["election"] = {"id": str(election.get("id")), "vote_day": _number(election.get("vote_day"), 0, 1e6),
                                  "nights_left": _number(election.get("nights_left"), 0, 100),
                                  "candidates": candidates,
                                  "tally": {str(k): _number(v, 0, 1e4) or 0 for k, v in tally.items()},
                                  "votes": {str(k): str(v) for k, v in votes.items()}}
        return result

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
        rules["foods"] = [f for f in raw.get("foods") or [] if f in FOODS] if isinstance(raw.get("foods"), list) else []
        costs_from = table(raw.get("move_costs"))
        rules["move_costs"] = {a: {b: _number(c, 0, 1000) for b, c in table(row).items() if _number(c, 0, 1000) is not None}
                               for a, row in costs_from.items()}
        staff_rules = table(raw.get("staff"))
        rules["staff"] = {"meals_per_employee": _number(staff_rules.get("meals_per_employee"), 1, 1000)
                          or MEALS_PER_EMPLOYEE,
                          "default_wage": _number(staff_rules.get("default_wage"), 0, 1000) or DEFAULT_WAGE}
        forecast = table(state.get("forecast"))
        cleaned_forecast = {"day": count(forecast.get("day")),
                            "expected_customers": clean_forecast(forecast.get("expected_customers"), rules["spots"]),
                            "weather": Agent._weather(forecast),
                            "events": [{"key": str(e.get("key") or ""), "name": str(e.get("name") or ""),
                                        "spot": e.get("spot")} for e in forecast.get("events") or []
                                       if isinstance(e, dict) and isinstance(e.get("spot"), str)]
                            if isinstance(forecast.get("events"), list) else []}
        for event, raw_event in zip(cleaned_forecast["events"], [e for e in forecast.get("events") or []
                                                                  if isinstance(e, dict) and isinstance(e.get("spot"), str)]
                                    if isinstance(forecast.get("events"), list) else []):
            event["opponent"] = str(raw_event.get("opponent") or "")
            event["text"] = str(raw_event.get("effect") or raw_event.get("text") or "")[:300]
        cleaned_forecast["diets"] = {spot: mix for spot, mix in table(forecast.get("diets")).items()
                                     if isinstance(mix, dict)}
        planned = {k: count(v) for k, v in table(forecast.get("planned")).items()}
        cleaned_forecast["planned"] = {k: v for k, v in planned.items() if v is not None} \
            if isinstance(forecast.get("planned"), dict) else None
        cleaned_forecast["riots"] = {field: forecast.get(field) for field in ("events", "warnings", "riots", "riot", "alerts")
                                     if forecast.get(field)}
        positions = {}
        for spot, info in table(state.get("spot_info")).items():
            info = info if isinstance(info, dict) else {}
            x, z = _number(info.get("x"), -1e5, 1e5), _number(info.get("z"), -1e5, 1e5)
            if x is not None and z is not None:
                positions[spot] = (x, z)
        you = table(state.get("you"))
        reputation = {k: _number(v, 0, 100) for k, v in table(you.get("reputation")).items()}
        trucks = [{"id": t.get("id"), "truck_name": str(t.get("truck_name") or ""),
                   "reputation_avg": _number(t.get("reputation_avg"), 0, 100),
                   "cash": _number(t.get("cash"), -1e9, 1e9),
                   "employees": _number(t.get("employees"), 0, 1000), "wage": _number(t.get("wage"), 0, 1000)}
                  for t in state.get("trucks") or [] if isinstance(t, dict) and isinstance(t.get("id"), str)] \
            if isinstance(state.get("trucks"), list) else []
        staff = table(you.get("staff"))
        labor = table(state.get("labor"))
        coupons = [c for c in you.get("coupons") or [] if isinstance(c, dict)] \
            if isinstance(you.get("coupons"), list) else []
        order = you.get("order") if isinstance(you.get("order"), dict) else None
        last_day = table(state.get("last_day"))
        moves = [m for m in last_day.get("moments") or [] if isinstance(m, dict)] \
            if isinstance(last_day.get("moments"), list) else []
        poached = [l for l in last_day.get("labor") or [] if isinstance(l, dict)] \
            if isinstance(last_day.get("labor"), list) else []
        politics = Agent._clean_politics(state.get("politics"))
        return {"mode": str(state.get("mode") or ""), "market": count(state.get("market")),
                "day": count(state.get("day")), "days": count(state.get("days")),
                "order_for_day": count(state.get("order_for_day")), "rules": rules,
                "features": features_of(state), "forecast": cleaned_forecast,
                "you": {"id": you.get("id") if isinstance(you.get("id"), str) else None,
                        "reputation": {k: v for k, v in reputation.items() if v is not None},
                        "cash": _number(you.get("cash"), -1e9, 1e9),
                        "order": order, "last_spot": you.get("last_spot") if isinstance(you.get("last_spot"), str) else None,
                        "vote": you.get("vote") if isinstance(you.get("vote"), str) else None,
                        "wrecked_until_day": count(you.get("wrecked_until_day")),
                        "staff": {k: _number(staff.get(k), 0, 1e6) for k in
                                  ("employees", "target", "wage", "capacity", "idle_days")} if staff else None,
                        "coupons": coupons},
                "trucks": trucks, "spot_positions": positions, "politics": politics,
                "labor": {k: _number(v, 0, 1e6) for k, v in labor.items()},
                "last_you": {"status": table(last_day.get("you")).get("status")},
                "poaching": [{"from": p.get("from"), "to": p.get("to"), "count": _number(p.get("count"), 0, 1000),
                              "wage": _number(p.get("wage"), 0, 1000)} for p in poached],
                "moments": [{"kind": m.get("kind"), "spot": m.get("spot")} for m in moves],
                "time": {"server": state.get("server_time"), "ends": state.get("phase_ends_at")}}

    def _crew(self, state, rules):
        """Tomorrow's crew, capacity and daily wages when staff is live: (crew, capacity, wage, employees)."""
        staff = (state.get("you") or {}).get("staff") or {}
        per = float((rules.get("staff") or {}).get("meals_per_employee") or MEALS_PER_EMPLOYEE)
        employees, target, wage = staff.get("employees"), staff.get("target"), staff.get("wage")
        if wage is None:
            wage = float((rules.get("staff") or {}).get("default_wage") or DEFAULT_WAGE)
        if employees is None and target is None:
            return None, None, wage, None
        pool = (state.get("labor") or {}).get("unemployed")
        # Each morning the crew is trimmed or refilled to the target; refilling needs unemployed workers.
        crew = employees if employees is not None else target
        if target is not None and employees is not None:
            crew = target if employees > target or pool is None or pool > 0 else employees
        return float(crew), float(crew) * per, float(wage), employees

    def plan(self, state, time_limit=1.0, laws=None, only=None, record=True):
        """Plan tonight's order: (order, report). `laws` replaces the laws in force (to judge a candidate in a
        vote), `only` limits the spots compared and `record=False` leaves the saved report alone."""
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
        if only:
            spots = [spot for spot in spots if spot in only] or spots
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
        game_day = any(_event_kind(e) == "gameday" for e in events)
        chances = self._weather(forecast)
        total_chance = sum(chances.values()) or 1.0
        mode, market = state.get("mode"), state.get("market")
        day = int(state.get("order_for_day") or (int(state.get("day") or 0) + 1))
        days_left = max(0, int(state.get("days") or day) - day)
        you = state.get("you") or {}
        own_rep = you.get("reputation") or {}
        positions = state.get("spot_positions") or {}
        metres = make_distance(positions)
        warnings = []

        # Laws in force, or a candidate's laws when judging a vote.
        in_force = law_policy(state["politics"]["laws"], game_day) if features["elections"] else law_policy([])
        policy = in_force if laws is None else law_policy(laws, game_day)

        def crowd_factor(spot):
            if laws is None:
                return 1.0 if FORECAST_HAS_LAWS else crowd_mult(policy, spot)
            return crowd_mult(policy, spot) / (crowd_mult(in_force, spot) if FORECAST_HAS_LAWS else 1.0)

        # Staff: wages on days we open, and sales (so meals) capped at employees x 50.
        crew = capacity = employees = None
        wages = 0.0
        if features["staff"]:
            crew, capacity, wage, employees = self._crew(state, rules)
            if crew is not None:
                wages = crew * wage
                if crew <= 0:
                    warnings.append("no staff: the truck cannot open")
                elif employees is not None and employees < crew:
                    warnings.append("{} of {} employees: hiring the rest tomorrow morning".format(
                        int(employees), int(crew)))
        # Coupons dropped on us pay half price for 20% of tomorrow's buyers each.
        unused = sum(1 for c in you.get("coupons") or [] if c.get("applied_day") is None)
        rev_mult = 1.0 - 0.5 * min(0.6, 0.2 * unused) if features["coupons"] else 1.0
        if rev_mult < 1.0:
            warnings.append("{} coupon drop(s) on us: tomorrow's revenue x {:.2f}".format(unused, rev_mult))
        riots = riot_warnings(forecast.get("riots") or {}, positions) if features["riots"] else []
        if riots:
            warnings.append("riot warning for {}".format(", ".join(sorted({r["spot"] for r in riots}))))
        planned = None
        if features["planned_counts"] and isinstance(forecast.get("planned"), dict):
            planned = {spot: count for spot, count in forecast["planned"].items() if spot in all_spots}
            own = (you.get("order") or {}).get("spot")
            if own in planned and planned[own] > 0:
                planned[own] -= 1  # the count includes our own order
        foods = (rules.get("foods") or list(FOODS)) if features["food"] else []
        foods = [f for f in FOODS if f in foods] or (["meat"] if features["food"] else [])

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
        lineups = self._lineups(profiles, all_spots, "where", LINEUP_SAMPLES, random.Random(seed), planned)
        by_id = {profile["id"]: profile for profile in profiles}
        counted = {spot: sum(len(lineup) * weight for lineup, weight in table.items())
                   for spot, table in lineups.items()}
        phantom = self.phantom_odds()
        if planned and sum(planned.values()) >= sum(sum(p["where"].values()) for p in profiles) - 0.5:
            phantom *= 0.5  # nearly every rival has ordered: a surprise newcomer is less likely
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
                    key = int(round(count * crowd_factor(spot)))
                    crowd_points[key] = crowd_points.get(key, 0.0) + chance / total_chance * weight
            diets = diet_mix(forecast.get("diets"), spot, game_day) if foods else None
            scenarios = self._scenarios(lineups[spot], self._condense(crowd_points), pop, spot, reps, by_id,
                                        values, temperature, policy, diets)
            model[spot] = {"scenarios": scenarios, "mu": mu, "sd": sd, "observations": observations,
                           "forecast": forecast_count, "diets": diets}

        walkers = self._walkers(model, lineups, all_spots, metres) if features["walk_over"] else {}

        rivals_report = {}
        per_spot = []
        best = None
        for spot in spots:
            entry = model[spot]
            scenarios, inflow = entry["scenarios"], walkers.get(spot, 0.0)
            chance_here = {p["name"]: p["where"].get(spot, 0.0) for p in profiles}
            rivals_report[spot] = {
                "expected": round(counted[spot] if planned else sum(chance_here.values()), 2),
                "likely": sorted(([name, round(chance, 2)] for name, chance in chance_here.items() if chance >= 0.2),
                                 key=lambda item: -item[1]),
                "customer_limit": round(entry["mu"], 2), "limit_spread": round(entry["sd"], 2),
                "observations": entry["observations"], "walk_in": round(inflow, 1),
            }

            move_cost = 0.0
            last_spot = you.get("last_spot")
            if features["moving_costs"] and last_spot and last_spot != spot:
                given = (rules.get("move_costs") or {}).get(last_spot, {}).get(spot)
                if given is None:
                    given = round(2.0 + 0.01 * metres(last_spot, spot))
                move_cost = float(given)
            fixed = wages + move_cost + policy["fixed"]
            rivals_report[spot]["move_cost"] = move_cost
            factor = rep_factor(float(own_rep.get(spot, 50)))

            def best_for(quality, food, scenarios=scenarios, inflow=inflow, fixed=fixed, factor=factor, spot=spot):
                qi = QUALITIES.index(quality)
                base_value = float(values.get(quality, DEFAULT_VALUES[quality]))
                cost = float(costs.get(quality, DEFAULT_COSTS[quality]))
                max_meals = int(budget // cost)
                if capacity is not None:
                    max_meals = min(max_meals, int(capacity))
                tax = meal_tax(policy, quality, food)
                top = min(price_max, base_value * factor - 0.01)
                low = price_min
                if policy["cap"] is not None:
                    top = min(top, policy["cap"])  # posting above the cap changes nothing
                if policy["floor"] is not None:
                    low = max(low, policy["floor"])

                def evaluate(price, value, fixed_meals=None):
                    charged = charged_price(price, policy)
                    points = self._points(scenarios, charged, value - charged, qi, temperature, inflow, food,
                                          capacity)
                    return self._best_meals(points, charged * rev_mult - tax, cost, max_meals, risk, salvage,
                                            fixed_meals, fixed, policy["top"])

                value = base_value * factor
                tried = {}
                step_price = max(low, math.ceil((cost + 0.25) * 4) / 4)
                coarse = []
                while step_price <= top:
                    coarse.append(round(step_price, 2))
                    step_price += 0.75
                best_row = None
                for price in coarse:
                    row = evaluate(price, value)
                    tried[price] = row
                    if row is not None and (best_row is None or row["objective"] > tried[best_row]["objective"]):
                        best_row = price
                if best_row is None:
                    return None
                # Refine around the best coarse price in quarter-dollar steps.
                for offset in (-0.5, -0.25, 0.25, 0.5):
                    price = round(best_row + offset, 2)
                    if low <= price <= top and price > cost and price not in tried:
                        tried[price] = evaluate(price, value)
                price = max((p for p in tried if tried[p] is not None), key=lambda p: tried[p]["objective"])
                candidate = dict(tried[price])
                rep_bonus = self._reputation_bonus(spot, quality, price, candidate, evaluate, base_value, own_rep,
                                                   days_left, state) * candidate["p_sell"]
                candidate.update(spot=spot, quality=quality, price=round(price, 2), food=food,
                                 rep_bonus=round(rep_bonus, 2), move_cost=move_cost,
                                 score=candidate["objective"] + rep_bonus + prefer.get(spot, 0.0))
                return candidate

            first_food = foods[0] if foods else None
            spot_best = None
            for quality in qualities:
                candidate = best_for(quality, first_food)
                if candidate is not None and (spot_best is None or candidate["score"] > spot_best["score"]):
                    spot_best = candidate
                if time.monotonic() - started > time_limit:
                    break
            if spot_best and len(foods) > 1:
                # Veggie-only customers only buy veggie, so try the other food on the best quality.
                other = best_for(spot_best["quality"], foods[1])
                if other is not None and other["score"] > spot_best["score"]:
                    spot_best = other
            if spot_best:
                spot_best["forecast"] = entry["forecast"]
                spot_best["riot_cost"] = 0.0
                if riots:
                    cost = riot_cost(riots, positions.get(spot), spot_best["expected_profit"], policy)
                    spot_best["riot_cost"] = cost
                    for field in ("expected_profit", "objective", "score"):
                        spot_best[field] -= cost
                per_spot.append(spot_best)
                if best is None or spot_best["score"] > best["score"]:
                    best = spot_best
            if time.monotonic() - started > time_limit:
                break

        staff_report = None
        if features["staff"]:
            staff_report = {"employees": employees, "crew_used": crew, "capacity": capacity, "wage": (you.get("staff") or {}).get("wage"), "wages": round(wages, 2),
                            "target": ((you.get("staff") or {}).get("target")),
                            "idle_days": ((you.get("staff") or {}).get("idle_days")),
                            "unemployed": (state.get("labor") or {}).get("unemployed"),
                            "poached": [p for p in state.get("poaching") or []
                                        if p.get("from") == you.get("id") or p.get("to") == you.get("id")]}
        skip = False
        if crew is not None and crew <= 0:
            fallback = best["spot"] if best else (spots[0] if spots else None)
            order = {"spot": fallback, "price": 10.0, "quality": "standard", "meals": 0}
            best = best or {"expected_sold": 0.0, "expected_profit": 0.0, "p_loss": 0.0, "expected_loss": 0.0}
        elif best is None or best["meals"] <= 0:
            return None, None
        else:
            idle = ((you.get("staff") or {}).get("idle_days")) or 0
            ordered_yesterday = (state.get("last_you") or {}).get("status") != "no_order"
            # A day that cannot cover its wages is better skipped, but never two in a row (the crew would leave).
            if features["staff"] and best["expected_profit"] < -5.0 and not idle and ordered_yesterday \
                    and laws is None:
                skip = True
                warnings.append("skipping tomorrow: expected profit {:.2f} after wages".format(best["expected_profit"]))
            order = {"spot": best["spot"], "price": best["price"], "quality": best["quality"],
                     "meals": int(best["meals"])}
            if features["food"]:
                order["food"] = best.get("food") or "meat"
            if capacity is not None and order["meals"] >= int(capacity) and \
                    int(budget // float(costs[order["quality"]])) > int(capacity):
                warnings.append("staff capacity {} limits meals below what the order could use".format(int(capacity)))
        politics = state.get("politics") or {}
        election = politics.get("election") or {}
        report = {
            "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "mode": mode, "market": market, "day": day, "order": order, "features": features,
            "expected_sold": round(best["expected_sold"], 1), "expected_profit": round(best["expected_profit"], 2),
            "p_loss": round(best["p_loss"], 3), "expected_loss": round(best["expected_loss"], 2),
            "risk_weight": round(risk, 2), "losing_streak": streak,
            "days_left": days_left, "surprise": round(self.surprise, 3), "phantom": round(phantom, 3),
            "forecast_bias": round(math.exp(ratio_mean), 3), "forecast_noise": round(ratio_sd, 3),
            "skip": skip, "warnings": warnings,
            "staff": staff_report,
            "politics": {"laws": policy["kinds"], "holder": (politics.get("office") or {}).get("holder"),
                         "crisis": (politics.get("office") or {}).get("crisis"),
                         "election": election.get("id"), "vote_day": election.get("vote_day"),
                         "vote": you.get("vote"), "tally": election.get("tally")},
            "planned": planned, "coupons_unused": unused, "riots": riots,
            "alternatives": [{field: (round(row[field], 2) if isinstance(row[field], float) else row[field])
                              for field in ("spot", "quality", "price", "meals", "expected_sold",
                                            "expected_profit", "p_loss", "rep_bonus", "score", "forecast",
                                            "move_cost", "food", "riot_cost") if field in row}
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
                         "extra_caution": caution or 0, "vote": settings.get("vote") or "auto"},
            "elapsed_ms": round((time.monotonic() - started) * 1000),
        }
        if record:
            self.last_report = report
            if self.persist:
                self._write_json(self.plan_path, report)
        return order, report

    # ---------- voting ----------

    @staticmethod
    def win_odds(candidates, tally, unvoted, ours, samples=400, turnout=0.7, seed=7):
        """Chance each candidate wins when our vote is `ours` (None: abstain) and the rest vote as follows.

        `tally` counts the votes already cast by others; each of `unvoted` other trucks still votes with
        chance `turnout`, for a candidate picked at random. Ties are settled by a draw."""
        rng = random.Random(seed)
        odds = {c: 0.0 for c in candidates}
        for _ in range(samples):
            counts = {c: float(tally.get(c, 0)) for c in candidates}
            for _n in range(unvoted):
                if rng.random() < turnout:
                    counts[rng.choice(candidates)] += 1
            if ours in counts:
                counts[ours] += 1
            top = max(counts.values())
            leaders = [c for c in candidates if counts[c] == top]
            for c in leaders:
                odds[c] += 1.0 / len(leaders) / samples
        return odds

    def vote_decision(self, state, time_limit=0.6):
        """Who to vote for during a campaign, or None to abstain: {"choice", "reason", "candidates", ...}.

        Each candidate's laws are tried on tonight's best few spots to see what they do to our daily profit.
        A vote only moves the result when it is close, and protesters target the trucks that voted for the
        office holder, so the default is to abstain unless a vote's expected gain clearly beats the protest cost."""
        started = time.monotonic()
        cleaned = self._clean_state(state)
        features = cleaned["features"]
        election = (cleaned["politics"] or {}).get("election")
        setting = self.settings().get("vote", "auto")
        info = {"choice": None, "reason": "no campaign", "candidates": [], "setting": setting}
        if not features["elections"] or not election or len(election["candidates"]) < 2:
            return info
        ids = [c["id"] for c in election["candidates"]]
        if setting == "abstain":
            info["reason"] = "settings: abstain"
            return info
        if setting in ids:
            info.update(choice=setting, reason="settings: candidate {}".format(setting))
            return info
        key = (cleaned["mode"], cleaned["market"], election["id"],
               None if cleaned["mode"] == "final" else tuple(sorted(election["tally"].items())))
        cached = self._vote_cache.get(key)
        if cached is not None:
            return dict(cached, setting=setting)
        you = cleaned["you"]
        # What each candidate's laws do to our expected daily profit, on the spots we would actually consider.
        _, base = self.plan(state, time_limit=time_limit / 3, record=False)
        if not base:
            info["reason"] = "no baseline plan"
            return info
        top = [row["spot"] for row in base["alternatives"][:3]]
        profit = {}
        for candidate in election["candidates"]:
            _, report = self.plan(state, time_limit=time_limit / 3, laws=candidate["policies"], only=top,
                                  record=False)
            profit[candidate["id"]] = report["expected_profit"] if report else 0.0
        day_profit = base["expected_profit"]
        revenue = base["expected_sold"] * float(base["order"]["price"])
        days = cleaned["days"] or 30
        term = max(0, min(TERM_DAYS, days - int(election["vote_day"] or cleaned["day"] or 0)))
        own = you["vote"] if you["vote"] in ids else None
        tally = dict(election["tally"])
        if own:
            tally[own] = max(0, tally.get(own, 0) - 1)
        unvoted = max(0, len(cleaned["trucks"]) - 1 - int(sum(tally.values())))
        abstain = self.win_odds(ids, tally, unvoted, None)
        cash = {t["id"]: t.get("cash") for t in cleaned["trucks"]}
        rows = []
        for candidate in election["candidates"]:
            cid = candidate["id"]
            voted = self.win_odds(ids, tally, unvoted, cid)
            value = term * sum((voted[k] - abstain[k]) * profit[k] for k in ids)
            # Protesters pick a voter of the office holder, usually the one that earned the most.
            voters = [t for t, v in election["votes"].items() if v == cid and t != you["id"]]
            richest = all((cash.get(t) or 0) <= (you.get("cash") or 0) for t in voters)
            others = unvoted * 0.7 / len(ids)
            v = len(voters) + others + 1
            p_target = 0.6 * (1.0 if richest else 0.1) / (1.0 + others) + 0.4 / v
            protest = term * 0.33 * p_target * PROTEST_SHARE * revenue
            net = value - voted[cid] * protest
            rows.append({"id": cid, "name": candidate["name"], "daily_profit": round(profit[cid], 2),
                         "p_win_if_abstain": round(abstain[cid], 3), "p_win_if_vote": round(voted[cid], 3),
                         "gain": round(value, 2), "protest_cost": round(voted[cid] * protest, 2),
                         "net": round(net, 2)})
        margin = max(20.0, 0.05 * term * max(0.0, day_profit))
        best = max(rows, key=lambda r: r["net"])
        info.update(candidates=rows, margin=round(margin, 2), term_days=term,
                    elapsed_ms=round((time.monotonic() - started) * 1000))
        if best["net"] > margin:
            info.update(choice=best["id"], reason="expected gain {:.0f} beats protest risk and margin".format(best["net"]))
        else:
            info["reason"] = "abstain: best net gain {:.0f} is under the margin {:.0f}".format(best["net"], margin)
        if len(self._vote_cache) > 50:
            self._vote_cache.clear()
        self._vote_cache[key] = dict(info)
        return info

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
    def _lineups(profiles, spots, key, samples, rng, planned=None):
        """Sample which rivals share each spot, and which of them cut their price on arriving.

        `planned` ({spot: orders in so far}, ours excluded) pins that many rivals to each spot, chosen by how
        likely they are to be there; the rivals still to order are placed as usual, fewer of them as orders
        come in."""
        lineups = {spot: {} for spot in spots}
        expected_total = sum(sum(p[key].values()) for p in profiles)

        def place(placed, profile, spot):
            cut = profile["cut_chance"] if profile["last"] and profile["last"] != spot else 0.0
            placed[spot].append((profile["id"], rng.random() < cut))

        for _ in range(samples):
            placed = {spot: [] for spot in spots}
            free = list(profiles)
            scale = 1.0
            if planned:
                for spot in spots:
                    for _n in range(min(int(planned.get(spot) or 0), len(free))):
                        weights = [max(1e-3, p[key].get(spot, 0.0)) for p in free]
                        roll = rng.random() * sum(weights)
                        index = 0
                        for index, weight in enumerate(weights):
                            roll -= weight
                            if roll <= 0:
                                break
                        place(placed, free.pop(index), spot)
                committed = sum(len(v) for v in placed.values())
                remaining = sum(sum(p[key].values()) for p in free)
                scale = min(1.0, max(0.0, expected_total - committed) / remaining) if remaining > 0 else 0.0
            for profile in free:
                roll = rng.random()
                for spot in spots:
                    roll -= profile[key].get(spot, 0.0) * scale
                    if roll < 0:
                        place(placed, profile, spot)
                        break
            for spot in spots:
                lineup = tuple(sorted(placed[spot]))
                lineups[spot][lineup] = lineups[spot].get(lineup, 0) + 1.0 / samples
        return lineups

    @staticmethod
    def _scenarios(lineups, crowd_points, pop, spot, reps, by_id, values, temperature, policy=None, diets=None):
        """Each likely rival lineup at each crowd size: (rivals, crowd tables, weight)."""
        kept = sorted(lineups.items(), key=lambda item: -item[1])[:MAX_LINEUPS]
        kept_weight = sum(weight for _, weight in kept) or 1.0
        tables = {count: crowd_tables(count, pop, diets) for count in crowd_points}
        scenarios = []
        for lineup, lineup_weight in kept:
            trucks = []
            for truck_id, cut in lineup:
                profile = by_id[truck_id]
                rep = (reps.get(truck_id) or {}).get(spot, 50.0)
                quality = profile["quality"]
                price = round(profile["price"] * (1.0 - profile["cut_size"]), 2) if cut else profile["price"]
                price = charged_price(price, policy) if policy else price
                value = float(values.get(quality, DEFAULT_VALUES[quality])) * rep_factor(rep) - price
                trucks.append({"id": truck_id, "price": price, "qi": QUALITIES.index(quality), "value": value,
                               "w": choice_weight(value, temperature), "meals": profile["meals"],
                               "food": profile.get("food") or "meat"})
            for count, crowd_weight in crowd_points.items():
                scenarios.append((trucks, tables[count], lineup_weight / kept_weight * crowd_weight, count))
        return scenarios

    @staticmethod
    def _walkers(model, lineups, spots, metres=distance):
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
                reach = WALK_BASE * math.exp(-metres(origin, spot) / WALK_SCALE)
                nearest = 1.0
                for other in spots:
                    if other not in (origin, spot) and metres(origin, other) < metres(origin, spot):
                        nearest *= 1.0 - occupied[other]
                total += hungry[origin] * reach * nearest
            inflow[spot] = total * WALK_ACCEPT
        return inflow

    @staticmethod
    def _points(scenarios, price, our_value, qi, temperature, inflow=0.0, food=None, capacity=None):
        """Meals we could sell in each scenario at this price (with walk-ins), with the scenario's weight."""
        if our_value < 0:
            return [(0.0, weight) for _, _, weight, _ in scenarios]
        ours = {"price": price, "qi": qi, "value": our_value, "w": choice_weight(our_value, temperature),
                "meals": 1e9, "food": food or "meat"}
        points = []
        for trucks, tables, weight, count in scenarios:
            sold, left = allocate(tables, trucks + [ours])
            demand = sold[-1]
            if inflow > 0:
                # Walkers arrive last and buy only what is left; we share them with rivals still in stock.
                others = sum(t["w"] for t, rest in zip(trucks, left) if rest > 0.5)
                groups = groups_of(tables)
                accepting = sum(_tail(group[q], price) for diet, group in groups if eats(diet, ours)
                                for q in range(qi + 1))
                total = sum(_tail(group[q], 0.0) for _, group in groups for q in range(3)) or 1.0
                demand += inflow * min(1.0, accepting / total) * ours["w"] / (ours["w"] + others)
            points.append((min(demand, capacity) if capacity is not None else demand, weight))
        return points

    def _reputation_bonus(self, spot, quality, price, candidate, evaluate, base_value, own_rep, days_left, state):
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
        raised = evaluate(price, base_value * rep_factor(higher), fixed_meals=candidate["meals"])
        if raised is None:
            return 0.0
        slope = (raised["objective"] - candidate["objective"]) / (higher - rep)
        stay = self.return_odds(state, [spot])[spot]
        # Reputation keeps its edge while we return (drift fades the rest), over the discounted days left.
        persist = REP_DISCOUNT * (1.0 - REP_DRIFT * (1.0 - stay))
        days = (1.0 - persist ** days_left) / (1.0 - persist)
        return REP_TRUST * stay * slope * gain * days

    @staticmethod
    def _best_meals(points, price, cost, max_meals, risk, salvage=0.0, fixed_meals=None, fixed=0.0, top=None):
        """Newsvendor choice of meals, cooked a little leaner when recent days lost money.

        `price` is what a sold meal nets (after coupons and per-meal taxes). With salvage, unsold meals
        refund part of their cost: the same as selling at a lower price for a lower effective food cost.
        `fixed` is the day's wages, moving cost and fees; `top` = (threshold, rate) taxes (or, if rate is
        negative, tops up) revenue above the threshold."""
        points = [(min(float(demand), float(max_meals)), weight) for demand, weight in points if weight > 0]
        total = sum(weight for _, weight in points)
        net_price, net_cost = price - salvage * cost, cost * (1.0 - salvage)
        if total <= 0 or net_price <= net_cost:
            return None
        points.sort()

        def choose(net_price):
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
            return best_m

        best_m = choose(net_price)
        if top and best_m > 0:
            # Revenue above the threshold is taxed: sell-outs there earn a little less per meal.
            share = sum(weight for demand, weight in points if price * min(best_m, demand) > top[0]) / total
            best_m = choose(net_price - top[1] * price * share)
        if fixed_meals is not None:
            best_m = int(fixed_meals)
        if best_m <= 0:
            return None
        sold_total = profit_total = loss_total = loss_chance = sell_chance = 0.0
        for demand, weight in points:
            sold = min(best_m, demand)
            profit = net_price * sold - net_cost * best_m - fixed
            if top:
                profit -= top[1] * max(0.0, price * sold - top[0])
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
