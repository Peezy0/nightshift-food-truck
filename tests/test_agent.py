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
        order, report = agent.plan(state)
        profit = {row["spot"]: row["expected_profit"] for row in report["alternatives"]}
        self.assertGreater(profit["thompson"], profit["oval"])
        self.assertEqual(order["spot"], "thompson")
        self.assertGreater(report["rivals"]["oval"]["expected"], 1.5)
        self.assertLess(report["rivals"]["thompson"]["expected"], 0.5)

    def test_fluid_model_sells_to_best_value_truck_first(self):
        pop = population(12.0, 2.8, (0.3, 0.5, 0.2))
        value = lambda quality, price: {"standard": 12, "premium": 16}[quality] * rep_factor(50) - price
        premium = {"price": 9.0, "qi": 2, "value": value("premium", 9.0), "meals": 200}
        standard = {"price": 9.0, "qi": 1, "value": value("standard", 9.0), "meals": 200}
        sold, _ = simulate_spot(100, pop, [standard, premium])
        self.assertGreater(sold[1], 50)
        # Premium food accepts every customer standard food does, so none are left for it.
        self.assertAlmostEqual(sold[0], 0.0, places=6)
        capped = dict(premium, meals=20)
        sold, _ = simulate_spot(100, pop, [standard, capped])
        self.assertAlmostEqual(sold[1], 20.0, places=6)
        self.assertGreater(sold[0], 0)

    def test_observe_records_history_once(self):
        agent = Agent(self.folder.name)
        state = make_state(1, {spot: 100 for spot in SPOTS}, crowd({"oval": RIVALS}))
        self.assertTrue(agent.observe(state))
        self.assertFalse(agent.observe(state))
        lines = (Path(self.folder.name) / "agent-history.jsonl").read_text().splitlines()
        self.assertEqual(len(lines), 1)
        self.assertEqual(json.loads(lines[0])["day"], 1)
        self.assertEqual(len(Agent(self.folder.name).days), 1)


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
