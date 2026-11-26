import json
import unittest
from npr_gm.sandbox import Sandbox,Source,Outcome,ContractError,encode,demo,parse_request

class SandboxTests(unittest.TestCase):
    def setUp(self):
        self.e=Sandbox(resources=4,clock=10,sources=[Source("public","Открытая запись."),Source("secret","Скрытый текст.",False)],
            outcomes=[Outcome("read",1,2,("public",),"Открытая запись прочитана."),Outcome("other",0,5,(),"Другой исход."),Outcome("expensive",5,0,(),"Недоступный исход.")])
        self.t=self.e.begin("Читаю запись.",["read"],budget=4)
    def candidate(self,**changes):
        c={"schema":"npr-gm-sandbox/1","turn_id":self.t.id,"revision":0,"outcome_id":"read","facts_used":["public"],"scene_draft":"Черновой текст."}
        c.update(changes);return encode(c)
    def read(self):
        return self.e.request(self.t,encode({"tool":"read_source","id":"public"}))
    def unchanged(self):
        self.assertEqual((self.e.resources,self.e.clock,self.e.revision),(4,10,0))
    def test_secret_not_in_context(self):
        self.assertNotIn("secret",encode(self.e.context(self.t)))
        self.assertNotIn("Скрытый текст",encode(self.e.context(self.t)))
    def test_uniform_unavailable(self):
        a=self.e.request(self.t,encode({"tool":"read_source","id":"secret"}))
        b=self.e.request(self.t,encode({"tool":"read_source","id":"missing"}))
        self.assertEqual(a,b);self.assertEqual(a,{"status":"unavailable"})
    def test_no_arbitrary_path(self):
        with self.assertRaises(ContractError): self.e.request(self.t,encode({"tool":"shell","id":"x"}))
        self.unchanged()
    def test_budget(self):
        for _ in range(4):self.read()
        with self.assertRaises(ContractError):self.read()
    def test_invalid_budget_attempt_counts(self):
        for _ in range(4):
            with self.assertRaises(ContractError):self.e.request(self.t,"bad")
        with self.assertRaisesRegex(ContractError,"tool_budget"):self.read()
    def test_duplicate_keys(self):
        with self.assertRaises(ContractError):parse_request('{"id":"a","id":"b"}')
    def test_no_unread_source(self):
        with self.assertRaises(ContractError):self.e.commit(self.t,self.candidate())
        self.unchanged()
    def test_no_hidden_citation(self):
        self.read()
        with self.assertRaises(ContractError):self.e.commit(self.t,self.candidate(facts_used=["public","secret"]))
        self.unchanged()
    def test_no_unapproved_action(self):
        self.read()
        with self.assertRaises(ContractError):self.e.commit(self.t,self.candidate(outcome_id="other"))
        self.unchanged()
    def test_no_model_cost_or_clock(self):
        self.read()
        for change in ({"clock":100},{"cost":0},{"delta":{}}):
            with self.assertRaises(ContractError):self.e.commit(self.t,self.candidate(**change))
        self.unchanged()
    def test_no_bool_revision(self):
        self.read()
        with self.assertRaises(ContractError):self.e.commit(self.t,self.candidate(revision=False))
        self.unchanged()
    def test_resource_gate(self):
        t=self.e.begin("Действие.",["expensive"])
        with self.assertRaisesRegex(ContractError,"insufficient_resource"):
            self.e.commit(t,self.candidate(turn_id=t.id,outcome_id="expensive",facts_used=[]))
        self.unchanged()
    def test_commit_authoritative(self):
        self.read()
        r=self.e.commit(self.t,self.candidate(scene_draft="Произвольный непроверенный текст."))
        self.assertEqual((r["resources"],r["clock"],r["revision"]),(3,12,1))
        self.assertEqual(r["player_text"],"Открытая запись прочитана.")
        self.assertNotIn("Произвольный",encode(r))
        self.assertEqual(self.e.receipt(self.t.id),r)
    def test_replay_and_concurrent_stale(self):
        other=self.e.begin("Ещё один запрос.",["other"])
        self.read();self.e.commit(self.t,self.candidate())
        with self.assertRaises(ContractError):self.e.commit(self.t,self.candidate())
        with self.assertRaises(ContractError):self.e.context(other)
        self.assertEqual((self.e.resources,self.e.clock,self.e.revision),(3,12,1))
    def test_demo(self):
        self.assertEqual(demo()["clock"],102)
