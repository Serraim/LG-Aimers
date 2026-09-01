import json
import unittest
from pathlib import Path

from scripts.build_s26_submission_model import validate_selection


REPO = Path(__file__).resolve().parent.parent


class BuildS26SubmissionModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload = json.loads(
            (REPO / "results" / "s26_slot_combination.json").read_text(encoding="utf-8")
        )

    def test_candidate_a_contract_and_exact_shift(self):
        selection, shift = validate_selection(self.payload, "A")
        self.assertEqual(selection["lgbm_slot"], ["cor5"])
        self.assertEqual(selection["cb_slot_z"], ["mcb", "e69"])
        self.assertAlmostEqual(selection["t"], 0.75)
        self.assertAlmostEqual(shift, -0.010629728763228485)

    def test_candidate_b_contract_and_exact_shift(self):
        selection, shift = validate_selection(self.payload, "B")
        self.assertEqual(selection["lgbm_slot"], ["cor5", "cal6"])
        self.assertEqual(selection["t"], 0.0)
        self.assertAlmostEqual(shift, -0.010213747947341034)

    def test_rejects_2024_selection(self):
        payload = dict(self.payload)
        payload["eval_season_used_in_selection"] = True
        with self.assertRaises(RuntimeError):
            validate_selection(payload, "A")


if __name__ == "__main__":
    unittest.main()
