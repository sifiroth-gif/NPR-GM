import copy
import unittest
from npr_gm.evaluate import check, load_cases, messages, generate

class EvalTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.cases=load_cases()
    def test_dataset(self):
        self.assertEqual(len(self.cases),24)
        self.assertEqual(len({x['group'] for x in self.cases}),12)
    def test_positive(self):
        for c in self.cases:
            with self.subTest(c=c['id']): self.assertEqual(check(c,c['chosen']),[])
    def test_negative(self):
        for c in self.cases:
            with self.subTest(c=c['id']): self.assertTrue(check(c,c['rejected']))
    def test_clock_bool(self):
        c=self.cases[0]; a=copy.deepcopy(c['chosen']); a['clock_advance']=True
        self.assertIn('clock_type',check(c,a))
    def test_missing_fact(self):
        c=self.cases[0]; a=copy.deepcopy(c['chosen']); a['facts_used']=[]
        self.assertIn('facts_used_missing',check(c,a))
    def test_duplicate(self):
        c=self.cases[0]; a=copy.deepcopy(c['chosen']); a['actions']*=2
        self.assertIn('actions_duplicate',check(c,a))
    def test_no_reference_leak(self):
        for c in self.cases:
            s=str(messages(c)); self.assertNotIn(c['chosen']['scene'],s); self.assertNotIn(c['rejected']['scene'],s)
    def test_remote_rejected(self):
        with self.assertRaises(ValueError): generate(self.cases[0],'https://example.com/v1/chat/completions','x',1)
    def test_invalid_schema(self): self.assertEqual(check(self.cases[0],None),['schema'])
