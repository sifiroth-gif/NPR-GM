import copy
import tempfile
import unittest
from pathlib import Path
from npr_gm.corpus import extract_session, review_export, private_output
from npr_gm.evaluate import load_cases

class CorpusTests(unittest.TestCase):
    def setUp(self):
        self.t=[{"id":"a","player":"Осматриваю витраж.","gm":"В центре изображён сад."},
                {"id":"b","player":"Как зовут художника?","gm":"В каталоге указан мастер.","kind":"ooc"}]
        self.d=[{"id":"a","d":{"hidden":"result"}},{"id":"b","d":None}]
        self.args=("opaque","a"*40,1,"archive/session-001-turns.jsonl","archive/deltas/session-001.jsonl")
    def test_candidate_not_train(self):
        r=extract_session(self.t,self.d,*self.args)
        self.assertEqual(len(r),2)
        self.assertFalse(r[0]["train_eligible"])
        self.assertIsNone(r[0]["context_snapshot"])
        self.assertIsNone(r[0]["chosen"])
        self.assertEqual(r[1]["prior_records_for_review"][0]["id"],"a")
        self.assertFalse(r[1]["delta_source"]["has_object"])
    def test_no_future_context(self):
        r=extract_session(self.t,self.d,*self.args)
        self.assertEqual(r[0]["prior_records_for_review"],[])
        self.assertNotIn("hidden",str(r[0]["context_snapshot"]))
    def test_missing_delta(self):
        r=extract_session(self.t,[],*self.args)
        self.assertIsNone(r[0]["delta_source"])
    def test_duplicate_rejected(self):
        with self.assertRaises(ValueError): extract_session(self.t+self.t,self.d,*self.args)
    def test_unpaired_delta(self):
        with self.assertRaises(ValueError): extract_session(self.t,self.d+[{"id":"c"}],*self.args)
    def test_private_path(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError): private_output(Path(d)/"public.jsonl")
            self.assertEqual(private_output(Path(d)/"private_data"/"x.jsonl").name,"x.jsonl")
    def reviewed(self):
        return {"id":"r","source":{"id":"test-only"},"review_status":"approved","train_eligible":True,
                "rights_status":"approved_for_training","semantic_leakage_review":"pass",
                "snapshot_review":"verified_as_of_turn","reviewer_id":"fixture","group_id":"g","split":"train",
                "applicable_ids":["C01","C02","C03","C04","C09"],"hard_results":dict.fromkeys(["C01","C02","C03","C04"],"pass"),
                "soft_scores":{"C09":3},"key_competence":"C09","context_snapshot":"Публичная выставка; открыт зал.",
                "player":"Расскажи об этом витраже.","chosen":"Ведущий описывает цветные стёкла и известного мастера."}
    def test_export_reviewed(self):
        r=review_export([self.reviewed()],{"g":"train"},load_cases())
        self.assertEqual([x["role"] for x in r[0]["messages"]],["system","user","assistant"])
    def test_pending_blocked(self):
        x=self.reviewed();x["review_status"]="pending"
        with self.assertRaises(ValueError): review_export([x],{"g":"train"},load_cases())
    def test_hard_fail(self):
        x=self.reviewed();x["hard_results"]["C02"]="unknown"
        with self.assertRaises(ValueError): review_export([x],{"g":"train"},load_cases())
    def test_rights_block(self):
        x=self.reviewed();x["rights_status"]="pending_review"
        with self.assertRaises(ValueError): review_export([x],{"g":"train"},load_cases())
    def test_public_overlap(self):
        x=self.reviewed();x["player"]=load_cases()[0]["player"]
        with self.assertRaises(ValueError): review_export([x],{"g":"train"},load_cases())
    def test_split_block(self):
        with self.assertRaises(ValueError): review_export([self.reviewed()],{"g":"holdout"},load_cases())
    def test_source_duplicate(self):
        x=self.reviewed();y=copy.deepcopy(x);y["chosen"]="Другой ответ.";y["id"]="s"
        with self.assertRaises(ValueError): review_export([x,y],{"g":"train"},load_cases())
