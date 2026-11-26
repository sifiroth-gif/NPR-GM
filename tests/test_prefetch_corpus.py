import copy
import json
import unittest
from npr_gm.prefetch_corpus import replay_prefetch,eligible_drafts,load_ledger,check_drafts
from npr_gm.structured import load_drafts
from npr_gm.sandbox import ContractError

class PrefetchCorpusTests(unittest.TestCase):
    def setUp(self):self.rows=load_drafts()
    def test_existing_candidates_only(self):
        r=check_drafts()
        self.assertEqual((r["drafts"],r["editorial_candidates"],r["held"],r["assistant_targets"],r["approved"]),(16,15,1,15,0))
        self.assertEqual(r["optional_tool_targets"],0)
    def test_required_are_host_input(self):
        m=replay_prefetch(self.rows[8])
        self.assertEqual([x["role"] for x in m],["system","user","assistant"])
        ctx=json.loads(m[1]["content"])
        self.assertNotIn("description",ctx["authorized_outcomes"][0])
        self.assertEqual({x["id"] for x in ctx["prefetched_sources"]},{"north_note","south_note"})
        self.assertEqual(json.loads(m[-1]["content"])["facts_used"],["north_note","south_note"])
    def test_missing_citation(self):
        r=copy.deepcopy(self.rows[8]);r["spec"]["reads"]=["north_note"]
        with self.assertRaises(ContractError):replay_prefetch(r)
    def test_optional_call_remains_target(self):
        r=copy.deepcopy(self.rows[0]);s=r["spec"]
        s["sources"].append({"id":"extra","text":"Additional synthetic fact","player_visible":True})
        s["authorized_sources"].append("extra");s["reads"].append("extra")
        m=replay_prefetch(r)
        self.assertEqual([x["role"] for x in m],["system","user","assistant","user","assistant"])
        self.assertEqual(json.loads(m[2]["content"])["id"],"extra")
        self.assertEqual(json.loads(m[3]["content"])["tool_result"]["id"],"extra")
    def test_hidden_optional_rejected(self):
        r=copy.deepcopy(self.rows[0]);s=r["spec"]
        s["sources"].append({"id":"secret","text":"private","player_visible":False})
        s["authorized_sources"].append("secret");s["reads"].append("secret")
        with self.assertRaises(ContractError):replay_prefetch(r)
    def test_review_coverage_and_hold(self):
        ledger=load_ledger()
        self.assertNotIn("T016",[r["id"] for r in eligible_drafts(self.rows,ledger)])
        with self.assertRaises(ValueError):eligible_drafts(self.rows,ledger[:-1])
