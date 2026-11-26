import copy
import unittest
from npr_gm.knowledge import load,validate
from npr_gm.knowledge_search import retrieve_for_model,retrieve

class KnowledgeRightsTests(unittest.TestCase):
    def setUp(self):self.registry,self.cards=load()
    def test_existing_drafts_all_blocked_from_model(self):
        for c in self.cards:
            self.assertEqual(retrieve_for_model(c["topic"],self.registry,self.cards),[])
    def test_human_editorial_path_remains_explicit(self):
        self.assertTrue(retrieve(self.cards[0]["topic"],self.cards,allow_draft=True))
        self.assertEqual(retrieve(self.cards[0]["topic"],self.cards),[])
    def test_source_denial_overrides_card_claim(self):
        c=copy.deepcopy(self.cards[0])
        c.update(status="reviewed_reference",model_use="approved_for_model_retrieval",permission_ref="fixture",expert_reviewer_id="fixture",review_date="2026-11-26")
        self.assertEqual(retrieve_for_model(c["topic"],self.registry,[c]),[])
    def test_affirmative_permissions_and_review_required(self):
        c=copy.deepcopy(self.cards[0]);r=copy.deepcopy(self.registry)
        source=next(s for s in r["sources"] if s["id"]==c["source_id"])
        source.update(model_use="approved_for_model_retrieval",permission_ref="synthetic test only")
        c.update(status="reviewed_reference",model_use="approved_for_model_retrieval",permission_ref="synthetic test only",expert_reviewer_id="fixture",review_date="2026-11-26")
        self.assertTrue(retrieve_for_model(c["topic"],r,[c]))
        for field in ("permission_ref","expert_reviewer_id","review_date"):
            test=copy.deepcopy(c);del test[field]
            self.assertEqual(retrieve_for_model(c["topic"],r,[test]),[])
    def test_draft_flag_tamper_detected(self):
        self.cards[0]["model_use"]="approved_for_model_retrieval"
        self.assertTrue(any("draft_model_use" in x for x in validate(self.registry,self.cards)))
