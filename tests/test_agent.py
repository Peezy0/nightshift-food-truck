"""Checks for the profit agent and its fallback in the bot."""

import json
from pathlib import Path
import tempfile
import unittest

from agent import Agent, population, rep_factor, simulate_spot
from bot import EventLog, choose_order, decide

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


if __name__ == "__main__":
    unittest.main()
