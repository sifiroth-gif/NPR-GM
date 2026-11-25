import copy
import unittest
from npr_gm.knowledge import load, validate, DOMAINS
class KnowledgeTest(unittest.TestCase):
    def setUp(self): self.registry, self.cards = load()
    def test_valid(self): self.assertEqual(validate(self.registry, self.cards), [])
    def test_initial_coverage(self):
        self.assertGreaterEqual(len(self.cards),18)
        for domain in DOMAINS: self.assertGreaterEqual(sum(c["domain"]==domain for c in self.cards),3)
    def test_training_block(self):
        self.cards[0]["train_eligible"] = True
        self.assertTrue(any("unreviewed_training" in x for x in validate(self.registry, self.cards)))
    def test_source_block(self):
        self.cards[0]["source_id"] = "missing"
        self.assertTrue(any("unknown_source" in x for x in validate(self.registry, self.cards)))
    def test_duplicate(self):
        self.cards.append(copy.deepcopy(self.cards[0]))
        self.assertIn("duplicate_card",validate(self.registry, self.cards))
