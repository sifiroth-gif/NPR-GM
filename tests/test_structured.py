import copy
import unittest
from npr_gm.structured import load_drafts,replay,validate_drafts,export_reviewed
from npr_gm.evaluate import load_cases
from npr_gm.sandbox import ContractError

class StructuredTests(unittest.TestCase):
    def setUp(self):self.rows=load_drafts()
    def test_all_replayed(self):
        r=validate_drafts(self.rows,load_cases())
        self.assertEqual((r["drafts"],r["groups"],r["assistant_targets"],r["approved"]),(16,16,32,0))
    def test_tool_result_not_assistant(self):
        m=replay(self.rows[8])
        self.assertEqual([x["role"] for x in m],["system","user","assistant","user","assistant","user","assistant"])
        self.assertIn("tool_result",m[3]["content"])
    def test_missing_source(self):
        r=copy.deepcopy(self.rows[8]);r["spec"]["reads"]=[]
        with self.assertRaises(ContractError):replay(r)
    def test_no_automatic_approval(self):
        with self.assertRaises(ValueError):export_reviewed(self.rows,{},load_cases())
    def approved(self):
        r=copy.deepcopy(self.rows[0]);r.update(review_status="approved",train_eligible=True,rights_status="approved_for_training",
            semantic_leakage_review="pass",snapshot_review="verified_as_of_turn",reviewer_id="fixture",split="train",
            applicable_ids=["C01","C02","C03","C04","C10"],hard_results=dict.fromkeys(["C01","C02","C03","C04","C10"],"pass"),soft_scores={"C10":3})
        return r
    def test_export_fixture(self):
        r=self.approved()
        self.assertEqual(len(export_reviewed([r],{r["group_id"]:"train"},load_cases())),1)
    def test_fail_gate(self):
        r=self.approved();r["hard_results"]["C10"]="unknown"
        with self.assertRaises(ValueError):export_reviewed([r],{r["group_id"]:"train"},load_cases())
    def test_split_gate(self):
        r=self.approved()
        with self.assertRaises(ValueError):export_reviewed([r],{r["group_id"]:"holdout"},load_cases())
