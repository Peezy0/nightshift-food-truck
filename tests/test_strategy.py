"""Regression checks for crowding risk in the live order strategy."""

import unittest

from bot import decide, estimate_rivals


class StrategyTests(unittest.TestCase):
    def setUp(self):
        self.state = {
            "forecast": {
                "expected_customers": {"union": 190, "northres": 67},
                "events": [{"spot": "union", "name": "Concert at the Union"}],
            },
            "last_day": {
                "spots": {
                    "union": {"trucks": [{"id": "rival-1"}]},
                    "northres": {"trucks": []},
                },
            },
            "you": {"id": "our-truck", "reputation": {"union": 50, "northres": 50}},
            "trucks": [{"id": "our-truck"}] + [{"id": f"rival-{number}"} for number in range(1, 7)],
            "rules": {"spots": ["union", "northres"], "budget": 500,
                      "costs": {"standard": 3.75}, "price_min": 1, "price_max": 30},
        }

    def test_concert_draws_rivals_even_if_last_day_was_quiet(self):
        self.assertEqual(estimate_rivals(self.state, "union"), 4)
        self.assertEqual(decide(self.state)["spot"], "northres")

    def test_forced_event_order_limits_food_at_risk(self):
        self.state["rules"]["spots"] = ["union"]
        order = decide(self.state)
        self.assertLessEqual(order["meals"], 25)

    def test_no_event_does_not_penalize_open_spot(self):
        self.state["forecast"]["events"] = []
        self.assertEqual(estimate_rivals(self.state, "northres"), 0)
        self.assertEqual(decide(self.state)["spot"], "union")


if __name__ == "__main__":
    unittest.main()
