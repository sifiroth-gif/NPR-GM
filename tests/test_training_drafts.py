import json
import unittest
from pathlib import Path
from npr_gm.evaluate import load_cases
from npr_gm.corpus import reference_strings, normalized, review_export

class DraftTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path=Path(__file__).resolve().parents[1]/"data/synthetic_train_draft.jsonl"
        cls.rows=[json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    def test_draft_count(self):
        self.assertEqual(len(self.rows),24)
        self.assertEqual(len({r["id"] for r in self.rows}),24)
    def test_unapproved(self):
        self.assertTrue(all(r["review_status"]=="pending" and r["train_eligible"] is False and r["split"]=="unassigned" for r in self.rows))
        with self.assertRaises(ValueError): review_export(self.rows,{},load_cases())
    def test_exact_overlap(self):
        blocked=reference_strings(load_cases())
        self.assertTrue(all(normalized(r["player"]) not in blocked and normalized(r["chosen"]) not in blocked for r in self.rows))
    def test_origin(self):
        self.assertTrue(all(r["source"]["kind"]=="original_synthetic" and r["context_snapshot"] and r["chosen"] for r in self.rows))
