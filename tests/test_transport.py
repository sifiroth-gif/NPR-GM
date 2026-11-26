import unittest
from npr_gm.transport import propose,local_exchange,sample
from npr_gm.sandbox import encode,ContractError

class TransportTests(unittest.TestCase):
    def setUp(self):self.e,self.t=sample()
    def candidate(self):
        return encode({"schema":"npr-gm-sandbox/1","turn_id":self.t.id,"revision":0,"outcome_id":"read_notice",
                       "facts_used":["catalog"],"scene_draft":"Зал закрыт."})
    def test_dialogue_read_then_proposal(self):
        inputs=[]
        answers=iter([encode({"tool":"read_source","id":"catalog"}),self.candidate()])
        def exchange(messages):
            inputs.append(messages)
            return next(answers)
        r=propose(self.e,self.t,exchange)
        self.assertEqual(r["tool_calls"],1)
        self.assertIn("Объявление: зал закрыт до утра.",inputs[1][-1]["content"])
        self.assertEqual((self.e.revision,self.e.clock),(0,100))
        self.assertIsNone(self.e.receipt(self.t.id))
        self.assertEqual(self.e.commit(self.t,r["candidate"])["clock"],102)
    def test_invalid_candidate_without_source(self):
        with self.assertRaises(ContractError):propose(self.e,self.t,lambda _:self.candidate())
        self.assertEqual(self.e.revision,0)
    def test_round_limit(self):
        with self.assertRaisesRegex(ContractError,"dialogue_round_limit"):
            propose(self.e,self.t,lambda _:encode({"tool":"read_source","id":"catalog"}),max_rounds=1)
    def test_remote_endpoint_rejected(self):
        for url in ("https://example.com/v1/chat/completions","http://localhost:8080/v1/chat/completions","http://127.0.0.1:8080/other"):
            with self.assertRaises(ValueError):local_exchange(url,"model")
    def test_unknown_tool_rejected(self):
        with self.assertRaises(ContractError):propose(self.e,self.t,lambda _:encode({"tool":"write_file","id":"x"}))
        self.assertEqual(self.e.revision,0)
