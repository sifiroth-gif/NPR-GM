import unittest
from npr_gm.knowledge import load
from npr_gm.knowledge_search import retrieve
from npr_gm.smoke import summarize
from npr_gm.hardware import probe

class PilotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.cards=load()[1]
    def test_default_excludes_drafts(self):
        self.assertEqual(retrieve("синаптическая передача",self.cards),[])
    def test_search(self):
        rows=retrieve("синаптическая передача",self.cards,"neurophysiology",allow_draft=True)
        self.assertEqual(rows[0]["id"],"K008")
        self.assertFalse(rows[0]["train_eligible"])
        self.assertTrue(rows[0]["url"].startswith("https://"))
    def test_domain_scope(self):
        self.assertEqual(retrieve("синаптическая передача",self.cards,"biology",scope="unknown",allow_draft=True),[])
    def test_no_hit(self):
        self.assertEqual(retrieve("zzzzqwerty",self.cards,allow_draft=True),[])
    def test_limit(self):
        with self.assertRaises(ValueError): retrieve("сон",self.cards,limit=0)
    def test_smoke_summary(self):
        s=summarize([{"seconds":2,"errors":[]},{"seconds":4,"errors":["schema"]},{"seconds":None,"errors":["transport:TimeoutError"]}])
        self.assertEqual((s["responses"],s["automatic_pass"],s["median_response_seconds"]),(2,1,3))
    def test_probe(self):
        p=probe()
        self.assertIn("physical_ram_bytes",p)
        self.assertNotIn("hostname",p)
