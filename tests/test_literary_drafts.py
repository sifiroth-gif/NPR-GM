import json
import unittest
from pathlib import Path
from npr_gm.corpus import normalized,reference_strings,review_export
from npr_gm.evaluate import load_cases

ROOT=Path(__file__).resolve().parents[1]/"data"
def rows(name):
    return [json.loads(x) for x in (ROOT/name).read_text(encoding="utf-8").splitlines() if x.strip()]

class LiteraryDraftTests(unittest.TestCase):
    def setUp(self):self.rows=rows("literary_train_draft.jsonl")
    def test_six_unique_groups_and_review(self):
        self.assertEqual(len(self.rows),6)
        self.assertEqual(len({r["group_id"] for r in self.rows}),6)
        ledger={x["id"]:x for x in rows("literary_train_review.jsonl")}
        self.assertEqual(set(ledger),{r["id"] for r in self.rows})
        for r in self.rows:
            self.assertEqual(ledger[r["id"]]["group_id"],r["group_id"])
            self.assertFalse(ledger[r["id"]]["train_eligible"])
    def test_draft_only(self):
        for r in self.rows:
            self.assertEqual(r["source"]["kind"],"original_synthetic")
            self.assertEqual(r["split"],"unassigned")
            self.assertFalse(r["train_eligible"])
            self.assertGreater(len(r["chosen"]),350)
        with self.assertRaises(ValueError):review_export(self.rows,{},load_cases())
    def test_exact_overlap(self):
        blocked=reference_strings(load_cases())
        for r in rows("synthetic_train_draft.jsonl"):
            blocked.update(normalized(r[k]) for k in ("player","chosen"))
        for r in self.rows:
            self.assertNotIn(normalized(r["player"]),blocked)
            self.assertNotIn(normalized(r["chosen"]),blocked)
