import copy
import json
import unittest
from npr_gm.optional_corpus import load_optional,check_optional
from npr_gm.prefetch_corpus import replay_prefetch
from npr_gm.structured import load_drafts
from npr_gm.sandbox import ContractError

class OptionalCorpusTests(unittest.TestCase):
    def setUp(self):self.rows=load_optional()
    def test_counts(self):
        r=check_optional()
        self.assertEqual((r["drafts"],r["assistant_targets"],r["optional_tool_targets"],r["zero_optional_calls"],r["approved"]),(8,15,7,2,0))
    def test_two_optional_results_host_generated(self):
        m=replay_prefetch(self.rows[2])
        self.assertEqual([x["role"] for x in m],["system","user","assistant","user","assistant","user","assistant"])
        ctx=json.loads(m[1]["content"])
        self.assertEqual([s["id"] for s in ctx["prefetched_sources"]],["seal_handover"])
        self.assertNotIn("решение общего собрания",m[1]["content"])
        self.assertEqual(json.loads(m[3]["content"])["tool_result"]["id"],"authority_clause")
        self.assertEqual(json.loads(m[5]["content"])["tool_result"]["id"],"effective_date")
    def test_no_distractor_read(self):
        for r in self.rows[-2:]:
            m=replay_prefetch(r)
            self.assertEqual([x["role"] for x in m],["system","user","assistant"])
    def test_missing_mandatory_citation(self):
        r=copy.deepcopy(self.rows[0]);r["spec"]["reads"]=["fitting_register"]
        with self.assertRaises(ContractError):replay_prefetch(r)
    def test_duplicate_group_rejected(self):
        r=copy.deepcopy(self.rows);r[0]["group_id"]=load_drafts()[0]["group_id"]
        with self.assertRaises(ValueError):check_optional(r)
    def test_approval_rejected(self):
        r=copy.deepcopy(self.rows);r[0]["train_eligible"]=True
        with self.assertRaises(ValueError):check_optional(r)
