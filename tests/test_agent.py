"""Checks for the profit agent and its fallback in the bot."""

import copy
import json
from pathlib import Path
import random
import tempfile
import time
import unittest

import agent as agent_module
from agent import Agent, population, rep_factor, simulate_spot
from bot import EventLog, choose_order, decide, maybe_vote, valid_order

SPOTS = ["oval", "thompson", "union", "stadium", "highst", "northres"]
RULES = {"budget": 500, "costs": {"cheap": 2.5, "standard": 3.75, "premium": 5.5},
         "values": {"cheap": 8, "standard": 12, "premium": 16}, "spots": SPOTS,
         "qualities": ["cheap", "standard", "premium"], "price_min": 1, "price_max": 30}
RIVALS = ["r1", "r2", "r3"]


def make_state(day, expected, last_spots=None, order_for_day=None):
    """A minimal API-shaped state: `day` days are complete, the forecast is for day + 1."""
    state = {
        "mode": "live", "market": 1, "markets": 1, "day": day, "days": 40, "phase": "night",
        "orders_open": True, "order_for_day": order_for_day or day + 1, "rules": RULES,
        "forecast": {"day": day + 1, "weather": {"clear": 1.0}, "events": [],
                     "crowds": {spot: "medium" for spot in SPOTS}, "expected_customers": expected},
        "you": {"id": "me", "truck_name": "Me", "cash": 0, "order": None,
                "reputation": {spot: 50 for spot in SPOTS}},
        "trucks": [{"id": "me", "truck_name": "Me", "reputation_avg": 50}]
                  + [{"id": rival, "truck_name": rival.upper(), "reputation_avg": 52} for rival in RIVALS],
    }
    if last_spots is not None:
        state["last_day"] = {"day": day, "weather": "clear", "events": [], "spots": last_spots, "you": {}}
    return state


def crowd(spot_trucks):
    """Last-day spot results: premium rivals at $10 with 90 meals, selling what a crowd leaves."""
    spots = {spot: {"customers": 100, "hungry": 100, "trucks": []} for spot in SPOTS}
    for spot, ids in spot_trucks.items():
        spots[spot]["trucks"] = [{"id": rival, "truck_name": rival.upper(), "price": 10.0,
                                  "quality": "premium", "meals": 90, "sold": 30} for rival in ids]
        spots[spot]["hungry"] = 100 - 30 * len(ids)
    return spots


class AgentTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)

    def test_plan_is_a_valid_order_within_limits(self):
        agent = Agent(self.folder.name, persist=False)
        state = make_state(0, {spot: 100 for spot in SPOTS})
        order, report = agent.plan(state)
        self.assertIn(order["spot"], SPOTS)
        self.assertIn(order["quality"], RULES["qualities"])
        self.assertGreaterEqual(order["price"], RULES["price_min"])
        self.assertLessEqual(order["price"], RULES["price_max"])
        self.assertIsInstance(order["meals"], int)
        self.assertGreater(order["meals"], 0)
        self.assertLessEqual(order["meals"] * RULES["costs"][order["quality"]], RULES["budget"])
        self.assertEqual(report["order"], order)
        self.assertEqual(len(report["alternatives"]), len(SPOTS))

    def test_avoids_spot_where_strong_rivals_are_sticky(self):
        agent = Agent(self.folder.name, persist=False)
        for day in (1, 2):
            agent.observe(make_state(day, {spot: 105 for spot in SPOTS},
                                     crowd({"oval": RIVALS})))
        state = make_state(2, {"oval": 106, "thompson": 104}, crowd({"oval": RIVALS}))
        state["rules"] = dict(RULES, spots=["oval", "thompson"])
        # With the market nearly over, reputation is not worth fighting three rivals for.
        state["days"] = 3
        order, report = agent.plan(state)
        profit = {row["spot"]: row["expected_profit"] for row in report["alternatives"]}
        self.assertGreater(profit["thompson"], profit["oval"])
        self.assertEqual(order["spot"], "thompson")
        self.assertGreater(report["rivals"]["oval"]["expected"], 1.5)
        self.assertLess(report["rivals"]["thompson"]["expected"], 0.5)

    def test_fair_choice_splits_customers_by_value_and_reroutes_sellouts(self):
        pop = population(12.0, 2.8, (0.3, 0.5, 0.2))
        value = lambda quality, price: {"standard": 12, "premium": 16}[quality] * rep_factor(50) - price
        premium = {"price": 9.0, "qi": 2, "value": value("premium", 9.0), "meals": 200}
        standard = {"price": 9.0, "qi": 1, "value": value("standard", 9.0), "meals": 200}
        sold, hungry = simulate_spot(100, pop, [standard, premium])
        # The better deal wins most customers, but the weaker one keeps a share.
        self.assertGreater(sold[1], 3 * sold[0])
        self.assertGreater(sold[0], 1)
        self.assertAlmostEqual(sum(sold) + hungry, 100.0, places=6)
        capped = dict(premium, meals=20)
        more, _ = simulate_spot(100, pop, [standard, capped])
        self.assertAlmostEqual(more[1], 20.0, places=6)
        self.assertGreater(more[0], sold[0])
        # Equal trucks split equally, and a higher price keeps a share instead of losing everyone.
        twin = simulate_spot(100, pop, [premium, dict(premium)])[0]
        self.assertAlmostEqual(twin[0], twin[1], places=6)
        dearer = dict(premium, price=11.0, value=value("premium", 11.0))
        self.assertGreater(simulate_spot(100, pop, [premium, dearer])[0][1], 5)

    def test_walk_over_and_salvage_change_the_plan(self):
        agent = Agent(self.folder.name, persist=False)
        state = make_state(0, {"oval": 150, "northres": 100})
        state["rules"] = dict(RULES, spots=["oval", "northres"])
        on, off = dict(state, features={"fair_choice": True, "walk_over": True, "salvage": True}), \
            dict(state, features={"fair_choice": True, "walk_over": False, "salvage": False})
        walk = agent.plan(on)[1]["rivals"]
        self.assertGreater(walk["oval"]["walk_in"], 0)
        self.assertEqual(agent.plan(off)[1]["rivals"]["oval"]["walk_in"], 0)
        # Salvage refunds part of the food cost, so it never lowers the profit we expect.
        self.assertGreaterEqual(agent.plan(on)[1]["expected_profit"], agent.plan(off)[1]["expected_profit"])

    def test_corrupted_days_are_repaired_or_dropped_not_learned_as_is(self):
        def spots_with(mutate):
            spots = crowd({"oval": RIVALS})
            mutate(spots)
            return spots

        agent = Agent(self.folder.name, persist=False)
        expected = {spot: 100 for spot in SPOTS}
        agent.observe(make_state(1, expected, spots_with(lambda s: s["oval"]["trucks"][0].update(sold=500))))
        agent.observe(make_state(2, expected, spots_with(lambda s: s["oval"]["trucks"][0].update(price="$9.25"))))
        agent.observe(make_state(3, expected, spots_with(lambda s: s["oval"].update(customers=13107))))
        agent.observe(make_state(4, expected, spots_with(lambda s: s["oval"]["trucks"][0].update(quality="gold"))))
        agent.observe(make_state(5, expected, spots_with(lambda s: s["oval"]["trucks"].append(dict(s["oval"]["trucks"][0])))))
        by_day = {record["day"]: record for record in agent.days}
        self.assertEqual(by_day[1]["spots"]["oval"]["trucks"][0]["sold"], 90)  # clipped to the meals cooked
        self.assertEqual(by_day[2]["spots"]["oval"]["trucks"][0]["price"], 9.25)
        self.assertNotIn("oval", by_day[3]["spots"])  # an absurd crowd is not learned
        self.assertEqual(len(by_day[4]["spots"]["oval"]["trucks"]), len(RIVALS) - 1)
        self.assertEqual(len(by_day[5]["spots"]["oval"]["trucks"]), len(RIVALS))
        self.assertEqual(len(agent.rejected), 4)  # a repaired money string is not an issue

    def test_plan_survives_bad_forecast(self):
        agent = Agent(self.folder.name, persist=False)
        state = make_state(0, {"oval": None, "union": 5000, "highst": 100, "thompson": 100, "stadium": "x",
                               "northres": 100})
        state["forecast"]["weather"] = {"clear": 0.71, "rain": None}
        order, report = agent.plan(state)
        self.assertIsNotNone(order)
        self.assertLess(report["rivals"]["union"]["walk_in"], 500)

    def test_observe_records_history_once(self):
        agent = Agent(self.folder.name)
        state = make_state(1, {spot: 100 for spot in SPOTS}, crowd({"oval": RIVALS}))
        self.assertTrue(agent.observe(state))
        self.assertFalse(agent.observe(state))
        lines = (Path(self.folder.name) / "agent-history.jsonl").read_text().splitlines()
        self.assertEqual(len(lines), 1)
        self.assertEqual(json.loads(lines[0])["day"], 1)
        self.assertEqual(len(Agent(self.folder.name).days), 1)


def day_with_us(day, spot, rivals_at=None, rival_price=9.0, our_price=9.0, sold=90):
    """A completed day where we sold premium at `spot` and each rival sat where `rivals_at` says."""
    spots = {s: {"customers": 100, "hungry": 20, "trucks": []} for s in SPOTS}
    spots[spot]["trucks"].append({"id": "me", "truck_name": "Me", "price": our_price,
                                  "quality": "premium", "meals": 90, "sold": sold})
    for rival, where in (rivals_at or {}).items():
        spots[where]["trucks"].append({"id": rival, "truck_name": rival.upper(), "price": rival_price,
                                       "quality": "premium", "meals": 90, "sold": 40})
    return spots


class ReputationStrategyTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)

    def contested_market(self, days_played, total_days=30):
        """We sold premium at Union every day for `days_played` days; two rivals camp there, one at Thompson."""
        agent = Agent(self.folder.name, persist=False)
        camp = {"r1": "union", "r2": "union", "r3": "thompson"}
        for day in range(1, days_played + 1):
            agent.observe(make_state(day, {spot: 100 for spot in SPOTS},
                                     day_with_us(day, "union", camp, rival_price=8.5)))
        state = make_state(days_played, {"union": 100, "thompson": 106},
                           day_with_us(days_played, "union", camp, rival_price=8.5))
        state["days"] = total_days
        state["mode"] = "live"
        state["you"]["reputation"]["union"] = min(100, 50 + 5 * days_played)
        # A spot that looks a little better today, but Union holds our reputation.
        state["rules"] = dict(RULES, spots=["union", "thompson"])
        return agent, state

    def test_home_spot_is_kept_across_days(self):
        for played in (4, 6, 8):
            agent, state = self.contested_market(played)
            order, report = agent.plan(state)
            self.assertEqual(order["spot"], "union", "after {} days".format(played))
        agent, state = self.contested_market(4)
        order, report = agent.plan(state)
        self.assertGreater(report["alternatives"][0]["rep_bonus"], 0)

    def test_reputation_is_worth_nothing_on_the_last_day(self):
        agent, state = self.contested_market(4, total_days=5)
        self.assertEqual(state["order_for_day"], 5)
        order, report = agent.plan(state)
        self.assertEqual({row["rep_bonus"] for row in report["alternatives"]}, {0.0})
        best_today = max(report["alternatives"], key=lambda row: row["expected_profit"])
        self.assertEqual(order["spot"], best_today["spot"])
        agent, state = self.contested_market(4, total_days=30)
        order, report = agent.plan(state)
        self.assertGreater(max(row["rep_bonus"] for row in report["alternatives"]), 0)

    def test_final_mode_is_risk_neutral(self):
        agent = Agent(self.folder.name, persist=False)
        losing = {spot: {"customers": 100, "hungry": 100, "trucks": []} for spot in SPOTS}
        losing["oval"]["trucks"] = [{"id": "me", "truck_name": "Me", "price": 8.0, "quality": "premium",
                                     "meals": 90, "sold": 10}]
        weights = {}
        for mode in ("live", "final"):
            for day in (1, 2, 3):
                state = make_state(day, {spot: 100 for spot in SPOTS}, losing)
                state["mode"] = mode
                agent.observe(state)
            state = make_state(3, {spot: 100 for spot in SPOTS}, losing)
            state["mode"] = mode
            weights[mode] = agent.plan(state)[1]["risk_weight"]
        self.assertGreater(weights["live"], 0.5)
        self.assertLessEqual(weights["final"], 0.05)

    def test_learns_that_rivals_chase_the_top_earner_and_undercut(self):
        agent = Agent(self.folder.name, persist=False)
        tops = ["union", "oval", "highst", "thompson", "union", "oval"]
        for day, top in enumerate(tops, 1):
            # We earn the most at `top`; r1 arrives there the next day with a lower price.
            where = {"r1": tops[day - 2] if day > 1 else "northres"}
            record = day_with_us(day, top, where, rival_price=10.0 - day, our_price=8.0)
            agent.observe(make_state(day, {spot: 100 for spot in SPOTS}, record))
        state = make_state(len(tops), {spot: 100 for spot in SPOTS},
                           day_with_us(len(tops), "oval", {"r1": "union"}, rival_price=4.0))
        profile = {p["id"]: p for p in agent.rival_profiles(state)}["r1"]
        self.assertGreater(profile["where"]["oval"], 0.3)
        self.assertGreater(profile["cut_chance"], 0.5)


LIVE_DAYS = Path(__file__).with_name("live_days_5_16.jsonl")


def live_state(day, union_rep):
    """The state the live bot saw before `day`, rebuilt from recorded live days (day 13 was a game day, day 16 had all rules live)."""
    rows = [json.loads(line) for line in LIVE_DAYS.read_text().splitlines()]
    names = {truck["id"]: truck["truck_name"] for row in rows for data in row["spots"].values()
             for truck in data["trucks"]}
    agent = Agent(tempfile.mkdtemp(), persist=False)
    for row in rows:
        if row["day"] < day:
            state = make_state(row["day"], {}, row["spots"])
            state["last_day"].update(weather=row["weather"], events=row["events"])
            state["you"]["id"] = row["you_id"]
            agent.observe(state)
    today = [row for row in rows if row["day"] == day][0]
    last = [row for row in rows if row["day"] == day - 1][0]
    state = make_state(day - 1, {spot: data["customers"] for spot, data in today["spots"].items()}, last["spots"])
    state["last_day"].update(weather=last["weather"], events=last["events"])
    state["you"] = {"id": today["you_id"], "truck_name": "Big Buck", "cash": 0, "order": None,
                    "reputation": dict({spot: 50 for spot in SPOTS}, union=union_rep)}
    state["trucks"] = [{"id": truck_id, "truck_name": names.get(truck_id, truck_id), "reputation_avg": 50}
                       for truck_id in today["rep_avg"]]
    return agent, state


class LiveRegressionTests(unittest.TestCase):
    def test_day_13_game_day_does_not_stay_at_a_crowded_union(self):
        # Alone at Union on days 11-12, we kept going and ranked third behind three rivals.
        agent, state = live_state(13, 68.4)
        state["forecast"]["events"] = [{"key": "gameday", "name": "Game day", "spot": "stadium"}]
        order, report = agent.plan(state)
        self.assertNotEqual(order["spot"], "union")
        self.assertGreaterEqual(report["phantom"], 0.1)
        # Even when forced to Union the price must leave room for a cheaper unseen rival.
        state["rules"] = dict(state["rules"], spots=["union"])
        forced, _ = agent.plan(state)
        self.assertLess(forced["price"], 11.4)

    def test_day_10_does_not_chase_into_thompson(self):
        # IBE moved to Thompson, where we had won, and undercut our premium.
        agent, state = live_state(10, 65.0)
        order, report = agent.plan(state)
        self.assertNotEqual(order["spot"], "thompson")


class ChooseOrderTests(unittest.TestCase):
    def setUp(self):
        self.state = make_state(3, {spot: 100 for spot in SPOTS}, crowd({"oval": RIVALS}))
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.log = EventLog(Path(folder.name) / "live.jsonl")

    def events(self):
        return [json.loads(line) for line in self.log.path.read_text().splitlines()]

    def test_falls_back_to_decide_when_plan_raises(self):
        class Broken:
            def plan(self, state, time_limit=1.0):
                raise RuntimeError("boom")

        order, report = choose_order(Broken(), self.state, self.log)
        self.assertEqual(order, decide(self.state))
        self.assertIsNone(report)
        error = self.events()[0]
        self.assertEqual((error["event"], error["error_type"]), ("agent_error", "RuntimeError"))

    def test_falls_back_when_plan_returns_none(self):
        class Silent:
            def plan(self, state, time_limit=1.0):
                return None, None

        self.assertEqual(choose_order(Silent(), self.state, self.log)[0], decide(self.state))

    def test_falls_back_when_order_breaks_the_rules(self):
        class Greedy:
            def plan(self, state, time_limit=1.0):
                return {"spot": "oval", "price": 10.0, "quality": "premium", "meals": 200}, {}

        order, report = choose_order(Greedy(), self.state, self.log)
        self.assertEqual(order, decide(self.state))
        self.assertIsNone(report)
        self.assertEqual(self.events()[0]["event"], "agent_error")

    def test_uses_agent_order_when_valid(self):
        agent = Agent(self.log.path.parent, persist=False)
        order, report = choose_order(agent, self.state, self.log)
        self.assertEqual(order, report["order"])


FUZZ_RUNS = 60
ALL_FEATURES = {"fair_choice": True, "walk_over": True, "salvage": True, "food": True, "moving_costs": True,
                "new_spots": True, "surprises": True, "football": True, "riots": True, "coupons": True,
                "staff": True, "planned_counts": True, "elections": True}
NEW_SPOTS = ["fisher", "kottman", "mirror", "medcenter", "lane", "rpac", "schott", "gateway"]
POSITIONS = {"oval": (100, 70), "thompson": (-12, 140), "union": (440, 388), "stadium": (-490, 48),
             "highst": (463, -186), "northres": (170, -478), "fisher": (-90, -456), "kottman": (-300, 200),
             "mirror": (250, 300), "medcenter": (50, 900), "lane": (0, -600), "rpac": (700, 500),
             "schott": (-600, -300), "gateway": (300, 1100)}


def policy(key, params=None, text=None):
    return {"key": key, "text": text or key, "params": params or {}}


def full_state(spots=None, **features):
    """A night with every new rule live and every new field present, for 14 open spots by default."""
    spots = spots or SPOTS + NEW_SPOTS
    state = make_state(5, {spot: 90 + 7 * i for i, spot in enumerate(spots)})
    state["rules"] = dict(RULES, spots=spots, foods=["meat", "veggie"], staff={"meals_per_employee": 50},
                          coupons={"cost": 20}, spot_names={s: s for s in spots},
                          move_costs={a: {b: 2 + abs(hash(a + b)) % 9 for b in spots if b != a} for a in spots})
    state["features"] = dict(ALL_FEATURES, **features)
    state["spot_info"] = {s: {"name": s, "x": POSITIONS[s][0], "z": POSITIONS[s][1], "region": "core",
                              "opened_day": 1} for s in spots}
    state["forecast"].update(
        crowds={s: "medium" for s in spots}, new_spots=[],
        diets={s: {"any": 0.7, "meat": 0.2, "veggie": 0.1} for s in spots},
        planned={s: 1 for s in spots[:5]},
        events=[{"key": "gameday", "name": "Game day vs Michigan", "opponent": "Michigan", "spot": "stadium",
                 "effect": "Riot warning: if Ohio State loses, fans riot on High Street"}])
    state["you"].update(
        reputation={s: 50 for s in spots}, last_spot="oval", vote="c2", wrecked_until_day=None,
        staff={"employees": 3, "target": 3, "wage": 11, "capacity": 150, "idle_days": 0},
        coupons=[{"day": 4, "applied_day": None}], coupons_dropped=[], news=[{"day": 4, "kind": "spot", "text": "x"}])
    state["labor"] = {"unemployed": 12, "lowest_asking_wage": 4, "median_asking_wage": 5.6}
    for truck in state["trucks"]:
        truck.update(employees=4, wage=6, cash=100, badge=None)
    state["politics"] = {
        "office": {"holder": {"id": "c1", "name": "Dean", "policies": [policy("price_cap", {"max_price": 12})]},
                   "since_day": 1, "until_day": 9, "crisis": False, "crisis_day": None},
        "election": {"id": "e2", "vote_day": 8, "nights_left": 2, "tally": {"c1": 1},
                     "votes": {"r1": "c1"},
                     "candidates": [{"id": "c1", "name": "A", "policies": [policy("meat_tax")]},
                                    {"id": "c2", "name": "B", "policies": [policy("veggie_tax")]},
                                    {"id": "c3", "name": "C", "policies": [policy("meat_tax"), policy("price_floor")]}]},
        "laws": [policy("price_cap", {"max_price": 12}), policy("luxury_tax", {"per_meal": {"premium": 0.5, "cheap": -0.5}})],
        "next_vote_day": 8, "history": []}
    state["last_day"] = {"day": 5, "weather": "clear", "events": [], "you": {"status": "played"},
                         "spots": crowd({"oval": RIVALS}),
                         "moments": [{"kind": "giveaway", "spot": "thompson", "trucks": [], "data": {}}],
                         "labor": [{"from": "r1", "to": "me", "count": 1, "wage": 12}]}
    for spot in spots:
        state["last_day"]["spots"].setdefault(spot, {"customers": 90, "hungry": 20, "trucks": []})
    return state


class NewRuleTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.agent = Agent(self.folder.name, persist=False)
        for day in (1, 2, 3):
            self.agent.observe(make_state(day, {spot: 100 for spot in SPOTS}, crowd({"oval": RIVALS})))

    def state(self, **features):
        state = make_state(3, {spot: 100 for spot in SPOTS}, crowd({"oval": RIVALS}))
        state["features"] = dict({"fair_choice": True, "walk_over": True, "salvage": True}, **features)
        return state

    def test_staff_caps_meals_and_wages_count(self):
        state = self.state(staff=True)
        state["you"]["staff"] = {"employees": 1, "target": 1, "wage": 11, "capacity": 50, "idle_days": 0}
        state["labor"] = {"unemployed": 5}
        order, report = self.agent.plan(state)
        self.assertLessEqual(order["meals"], 50)
        self.assertEqual(report["staff"]["capacity"], 50)
        self.assertTrue(any("capacity" in w for w in report["warnings"]))
        free = self.agent.plan(self.state())[1]
        state["you"]["staff"].update(employees=3, target=3, capacity=150, wage=30)
        paid = self.agent.plan(state)[1]
        self.assertAlmostEqual(free["expected_profit"] - paid["expected_profit"], 90.0, delta=1.0)
        self.assertEqual(paid["staff"]["wages"], 90.0)

    def test_skips_a_day_that_cannot_cover_wages_but_never_two_in_a_row(self):
        state = self.state(staff=True)
        state["you"]["staff"] = {"employees": 10, "target": 10, "wage": 60, "capacity": 500, "idle_days": 0}
        state["labor"] = {"unemployed": 5}
        order, report = self.agent.plan(state)
        self.assertTrue(report["skip"])
        self.assertIsNone(choose_order(self.agent, state)[0])
        state["you"]["staff"]["idle_days"] = 1  # a second day with no order would cost the crew
        order, report = self.agent.plan(state)
        self.assertFalse(report["skip"])
        self.assertGreater(order["meals"], 0)

    def test_no_staff_sends_a_zero_meal_order(self):
        state = self.state(staff=True)
        state["you"]["staff"] = {"employees": 0, "target": 0, "wage": 11, "capacity": 0, "idle_days": 0}
        order, report = self.agent.plan(state)
        self.assertEqual(order["meals"], 0)
        self.assertTrue(any("no staff" in w for w in report["warnings"]))

    def test_price_cap_and_luxury_tax(self):
        state = self.state(elections=True)
        free = self.agent.plan(state)[1]
        state["politics"] = {"laws": [policy("price_cap", {"max_price": 8})]}
        order, report = self.agent.plan(state)
        self.assertLessEqual(order["price"], 8.0)
        self.assertLess(report["expected_profit"], free["expected_profit"])
        state["politics"] = {"laws": [policy("luxury_tax", {"per_meal": {"premium": 0.5, "cheap": -0.5}})]}
        taxed, report = self.agent.plan(state)
        premium_free = [a for a in free["alternatives"] if a["quality"] == "premium"]
        premium_taxed = [a for a in report["alternatives"] if a["quality"] == "premium"]
        if premium_free and premium_taxed:
            self.assertLess(max(a["expected_profit"] for a in premium_taxed),
                            max(a["expected_profit"] for a in premium_free))
        self.assertIn("luxury_tax", report["politics"]["laws"])

    def test_law_policy_reads_params_and_falls_back_to_published_numbers(self):
        pol = agent_module.law_policy([policy("price_floor", {"min_price": 9}), policy("junk_food_tax"),
                                       policy("tax_the_top"), policy("curfew")])
        self.assertEqual(pol["floor"], 9.0)
        self.assertEqual(agent_module.charged_price(7.0, pol), 9.0)
        self.assertEqual(pol["tax_quality"], {"cheap": 0.5, "premium": -0.5})
        self.assertEqual(pol["top"], (500.0, 0.1))
        self.assertEqual(pol["fixed"], -20.0)
        self.assertEqual(agent_module.crowd_mult(pol, "highst"), 0.75)
        self.assertEqual(agent_module.crowd_mult(pol, "oval"), 1.1)

    def election(self, candidates, votes=None, tally=None, trucks=None):
        state = self.state(elections=True)
        state["politics"] = {"office": None, "laws": [], "election": {
            "id": "e{}".format(len(candidates)) + str(random.random()), "vote_day": 8, "nights_left": 2,
            "candidates": [{"id": cid, "name": cid, "policies": pols} for cid, pols in candidates],
            "tally": tally or {}, "votes": votes or {}}}
        if trucks:
            state["trucks"] = state["trucks"][:trucks]
        return state

    def test_vote_abstains_when_gains_are_small(self):
        state = self.election([("c1", [policy("curfew")]), ("c2", [policy("quiet_core")]), ("c3", [])])
        decision = self.agent.vote_decision(state)
        self.assertIsNone(decision["choice"])
        self.assertIn("abstain", decision["reason"])
        self.assertEqual(len(decision["candidates"]), 3)

    def test_vote_goes_to_a_clearly_better_candidate(self):
        rebate = [policy("luxury_tax", {"per_meal": {"premium": -3.0, "cheap": 3.0}})]
        state = self.election([("c1", rebate), ("c2", []), ("c3", [])], trucks=2)
        decision = self.agent.vote_decision(state)
        self.assertEqual(decision["choice"], "c1")
        # The team can override: abstain, or a named candidate.
        path = Path(self.folder.name) / "settings.json"
        for setting, expected in (("abstain", None), ("c3", "c3")):
            path.write_text(json.dumps({"vote": setting}))
            tuned = Agent(self.folder.name, persist=False, settings_path=path)
            self.assertEqual(tuned.vote_decision(state)["choice"], expected)

    def test_bot_votes_only_when_the_desired_vote_differs(self):
        rebate = [policy("luxury_tax", {"per_meal": {"premium": -3.0, "cheap": 3.0}})]
        state = self.election([("c1", rebate), ("c2", []), ("c3", [])], trucks=2)
        log = EventLog(Path(self.folder.name) / "live.jsonl")
        sent, voted = [], {}
        maybe_vote(self.agent, state, log, lambda body: sent.append(body) or {"ok": True}, voted)
        maybe_vote(self.agent, state, log, lambda body: sent.append(body) or {"ok": True}, voted)
        self.assertEqual(sent, [{"candidate": "c1"}])
        state["you"]["vote"] = "c1"
        maybe_vote(self.agent, state, log, lambda body: sent.append(body) or {"ok": True}, {})
        self.assertEqual(len(sent), 1)

        def failing(body):
            raise OSError("down")

        maybe_vote(self.agent, state, log, failing, {})  # a failing vote never raises

    def test_planned_counts_shift_the_lineup(self):
        state = self.state(planned_counts=True)
        none = self.agent.plan(state)[1]
        # All three rivals have already ordered for Thompson: the Oval is theirs to leave alone.
        state["forecast"]["planned"] = {spot: 0 for spot in SPOTS}
        state["forecast"]["planned"]["thompson"] = 3
        elsewhere = self.agent.plan(state)[1]
        profit = lambda report, spot: [a for a in report["alternatives"] if a["spot"] == spot][0]["expected_profit"]
        self.assertAlmostEqual(elsewhere["rivals"]["thompson"]["expected"], 3.0, delta=0.15)
        self.assertAlmostEqual(elsewhere["rivals"]["oval"]["expected"], 0.0, delta=0.15)
        self.assertGreater(profit(elsewhere, "oval"), profit(none, "oval"))
        self.assertLess(profit(elsewhere, "thompson"), profit(none, "thompson"))
        # Our own order is in the count and is not a rival.
        state["you"]["order"] = {"spot": "thompson", "price": 9, "quality": "standard", "meals": 50}
        self.assertAlmostEqual(self.agent.plan(state)[1]["rivals"]["thompson"]["expected"], 2.0, delta=0.15)

    def test_new_spot_has_priors_and_dynamic_positions(self):
        state = self.state(new_spots=True)
        state["rules"] = dict(RULES, spots=SPOTS + ["fisher", "medcenter"])
        state["forecast"]["expected_customers"].update(fisher=200, medcenter=60)
        state["spot_info"] = {"fisher": {"x": -90, "z": -456}, "medcenter": {"x": 50, "z": 900},
                              "oval": {"x": 100, "z": 70}}
        order, report = self.agent.plan(state)
        self.assertEqual(report["rivals"]["fisher"]["customer_limit"], 14.0)
        self.assertIn("medcenter", report["rivals"])
        metres = agent_module.make_distance(agent_module.Agent._clean_state(state)["spot_positions"])
        self.assertAlmostEqual(metres("oval", "fisher"), 559.2, delta=1)
        self.assertEqual(metres("oval", "thompson"), 130)  # no positions: published table
        self.assertEqual(order["spot"], "fisher")

    def test_move_cost_keeps_a_marginal_move_home(self):
        state = self.state(moving_costs=True)
        state["you"]["last_spot"] = "oval"
        state["rules"] = dict(RULES, move_costs={"oval": {s: 40 for s in SPOTS if s != "oval"}})
        report = self.agent.plan(state)[1]
        by_spot = {a["spot"]: a for a in report["alternatives"]}
        self.assertEqual(by_spot["oval"]["move_cost"], 0)
        self.assertEqual(by_spot["union"]["move_cost"], 40.0)
        free = self.agent.plan(self.state())[1]
        free_by = {a["spot"]: a["expected_profit"] for a in free["alternatives"]}
        self.assertAlmostEqual(free_by["union"] - by_spot["union"]["expected_profit"], 40.0, delta=1.0)
        # First day out (no last spot) is free.
        state["you"]["last_spot"] = None
        self.assertEqual({a["move_cost"] for a in self.agent.plan(state)[1]["alternatives"]}, {0.0})

    def test_meat_or_veggie_follows_the_diets(self):
        state = self.state(food=True)
        state["rules"] = dict(RULES, foods=["meat", "veggie"])
        state["forecast"]["diets"] = {s: {"any": 0.3, "meat": 0.1, "veggie": 0.6} for s in SPOTS}
        self.assertEqual(self.agent.plan(state)[0]["food"], "veggie")
        state["forecast"]["diets"] = {s: {"any": 0.3, "meat": 0.6, "veggie": 0.1} for s in SPOTS}
        self.assertEqual(self.agent.plan(state)[0]["food"], "meat")
        state["features"]["food"] = False
        self.assertNotIn("food", self.agent.plan(state)[0])
        self.assertFalse(valid_order({"spot": "oval", "price": 9, "quality": "standard", "meals": 5, "food": "fish"},
                                     dict(RULES, foods=["meat", "veggie"])))

    def test_riot_warning_steers_away_from_the_riot(self):
        state = self.state(riots=True, football=True)
        state["spot_info"] = {s: {"x": POSITIONS[s][0], "z": POSITIONS[s][1]} for s in SPOTS}
        state["forecast"]["events"] = [{"key": "gameday", "name": "Game day vs Michigan", "opponent": "Michigan",
                                        "spot": "stadium", "effect": "Riot warning: fans riot on High Street if we lose"}]
        report = self.agent.plan(state)[1]
        cost = {a["spot"]: a["riot_cost"] for a in report["alternatives"]}
        self.assertEqual(max(cost, key=cost.get), "highst")
        self.assertGreater(cost["highst"], 5 * cost["union"])
        self.assertTrue(any("riot" in w for w in report["warnings"]))

    def test_coupons_cost_revenue_and_are_never_dropped(self):
        state = self.state(coupons=True)
        free = self.agent.plan(state)[1]
        state["you"]["coupons"] = [{"day": 3, "applied_day": None}]
        report = self.agent.plan(state)[1]
        self.assertLess(report["expected_profit"], free["expected_profit"])
        self.assertEqual(report["coupons_unused"], 1)
        source = Path(agent_module.__file__).read_text() + (Path(agent_module.__file__).with_name("bot.py").read_text())
        self.assertNotIn("/coupons", source)  # we never drop coupons on anyone

    def test_day_23_late_night_rush_with_rain_and_wages(self):
        # Live day 23: rain hit the Oval (26 customers of a forecast 50) and wages made a thin day a loss.
        state = self.state(staff=True, moving_costs=True)
        state["forecast"]["expected_customers"] = {"oval": 50, "thompson": 37, "union": 48, "stadium": 20,
                                                   "highst": 237, "northres": 142}
        state["forecast"]["events"] = [{"key": "latenight", "name": "Late-night rush", "spot": "highst"}]
        state["forecast"]["weather"] = {"clear": 0.61, "cloudy": 0.27, "rain": 0.12}
        state["you"]["staff"] = {"employees": 3, "target": 3, "wage": 11, "capacity": 150, "idle_days": 0}
        state["you"]["last_spot"] = "oval"
        state["labor"] = {"unemployed": 10}
        state["rules"] = dict(RULES, move_costs={"oval": {s: 6 for s in SPOTS if s != "oval"}})
        order, report = self.agent.plan(state)
        # The event is already in the forecast counts: they are used as given, not scaled again.
        self.assertEqual({a["spot"]: a["forecast"] for a in report["alternatives"]}["highst"], 237.0)
        self.assertEqual(report["staff"]["wages"], 33.0)
        by_spot = {a["spot"]: a for a in report["alternatives"]}
        self.assertLess(by_spot["oval"]["expected_profit"], 0.2 * by_spot["highst"]["expected_profit"])
        self.assertIn(order["spot"], ("highst", "northres"))
        # More rain at the outdoor Oval means fewer meals there.
        state["forecast"]["weather"] = {"clear": 0.1, "cloudy": 0.1, "rain": 0.8}
        rainy = {a["spot"]: a for a in self.agent.plan(state)[1]["alternatives"]}
        self.assertLess(rainy["oval"]["expected_profit"], by_spot["oval"]["expected_profit"])

    def test_every_new_field_present(self):
        state = full_state()
        self.assertTrue(self.agent.observe(state))
        started = time.monotonic()
        order, report = self.agent.plan(state)
        self.assertLess(time.monotonic() - started, 0.3)
        self.assertTrue(valid_order(order, state["rules"]))
        self.assertIn(order["food"], ("meat", "veggie"))
        self.assertEqual(report["staff"]["capacity"], 150)
        self.assertIn("price_cap", report["politics"]["laws"])
        self.assertEqual(report["politics"]["election"], "e2")
        self.assertIn("coupon", " ".join(report["warnings"]))
        self.assertIsNotNone(self.agent.vote_decision(state)["reason"])

    def test_fuzzed_states_never_raise(self):
        rng = random.Random(11)
        junk = [None, "x", -5, 1e12, [], {}, [1, "a"], {"a": None}, float("nan"), True, "$9.5"]

        def mutate(node, depth=0):
            if isinstance(node, dict):
                for key in list(node):
                    roll = rng.random()
                    if roll < 0.06:
                        node[key] = rng.choice(junk)
                    elif roll < 0.1:
                        del node[key]
                    else:
                        mutate(node[key], depth + 1)
            elif isinstance(node, list):
                for index in range(len(node)):
                    if rng.random() < 0.06:
                        node[index] = rng.choice(junk)
                    else:
                        mutate(node[index], depth + 1)

        template = full_state()
        for _ in range(FUZZ_RUNS):
            state = copy.deepcopy(template)
            mutate(state)
            agent = Agent(self.folder.name, persist=False)
            agent.observe(state)
            try:
                order, report = agent.plan(state)
                agent.vote_decision(state)
            except Exception as error:  # report which mutation broke it
                self.fail("plan or vote raised {!r} on {}".format(error, json.dumps(state, default=str)[:300]))
            if order is not None:
                self.assertIn(order["quality"], QUALITIES_ALL)
                self.assertIsInstance(order["meals"], int)


QUALITIES_ALL = ("cheap", "standard", "premium")


if __name__ == "__main__":
    unittest.main()
