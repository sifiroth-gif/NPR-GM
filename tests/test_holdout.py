import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from npr_gm.holdout import load_private
from npr_gm.evaluate import messages
class HoldoutTests(unittest.TestCase):
    def case(self):
        return {"id":"fixture","group":"g","split":"holdout","source":"original_synthetic","context":"Демонстрационный макет.","player":"Покажите макет, пожалуйста.","manual_criteria":["private criteria"],"allowed_facts":["a"],"required_facts":["a"],"allowed_actions":[],"required_actions":[],"clock_min":0,"clock_max":1,"forbidden_text":[]}
    def test_checksum_and_no_criteria_in_prompt(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"private_data"/"fixture.jsonl";p.parent.mkdir()
            p.write_text(json.dumps(self.case())+"\n",encoding="utf-8")
            h=hashlib.sha256(p.read_bytes()).hexdigest()
            rows=load_private(p,h)
            self.assertEqual(len(rows),1)
            self.assertNotIn("private criteria",str(messages(rows[0])))
            with self.assertRaises(ValueError): load_private(p,"0"*64)
    def test_public_path_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"fixture.jsonl";p.write_text("{}")
            with self.assertRaises(ValueError): load_private(p,"0"*64)
    def test_bad_id_relation(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"private_data"/"fixture.jsonl";p.parent.mkdir()
            r=self.case();r["required_facts"]=["b"];p.write_text(json.dumps(r)+"\n")
            with self.assertRaises(ValueError): load_private(p,hashlib.sha256(p.read_bytes()).hexdigest())
    def test_review_count_and_gate(self):
        p=Path(__file__).resolve().parents[1]/"data/synthetic_train_review.jsonl"
        rows=[json.loads(s) for s in p.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(rows),24)
        self.assertEqual(sum(x["decision"]=="candidate" for x in rows),15)
        self.assertTrue(all(x["train_eligible"] is False for x in rows))
