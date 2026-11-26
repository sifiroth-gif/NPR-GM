import json
import unittest
from npr_gm.prefetch_experiment import CASES, fixture, run_arm, experiment
from npr_gm.sandbox import encode

def candidate(turn, facts):
    return encode({"schema":"npr-gm-sandbox/1","turn_id":turn.id,"revision":turn.revision,
                   "outcome_id":"o1","facts_used":facts,"scene_draft":"Черновик."})

class PrefetchExperimentTests(unittest.TestCase):
    def test_prefetch_does_not_bypass_explicit_citation(self):
        for facts,status in [([],"missing_required_citation"),(["a17"],"valid"),
                             (["a17","hidden"],"unread_or_extra_citation")]:
            e,t=fixture(CASES[0])
            r=run_arm(e,t,lambda _:candidate(t,facts),prefetch=True)
            self.assertEqual(r["status"],status)
            self.assertTrue(r["unchanged"])

    def test_direct_must_actually_read(self):
        e,t=fixture(CASES[0])
        r=run_arm(e,t,lambda _:candidate(t,["a17"]),prefetch=False)
        self.assertEqual(r["status"],"unread_or_extra_citation")
        e,t=fixture(CASES[0])
        answers=iter([encode({"tool":"read_source","id":"a17"}),candidate(t,["a17"])])
        self.assertEqual(run_arm(e,t,lambda _:next(answers),prefetch=False)["status"],"valid")

    def test_redacted_descriptions_and_no_secret(self):
        for mode in (False,True):
            e,t=fixture(CASES[1])
            def exchange(messages):
                ctx=json.loads(messages[1]["content"])
                self.assertNotIn("description",ctx["authorized_outcomes"][0])
                self.assertNotIn("SECRET_SENTINEL",encode(ctx))
                self.assertNotIn("UNRELATED_SENTINEL",encode(ctx))
                self.assertEqual(len(ctx["prefetched_sources"]),2 if mode else 0)
                self.assertEqual("Мастер вернётся в четверг." in encode(ctx),mode)
                return candidate(t,["b21","b22"])
            run_arm(e,t,exchange,prefetch=mode)

    def test_prefetch_consumes_same_budget(self):
        e,t=fixture(CASES[1])
        r=run_arm(e,t,lambda _:encode({"tool":"read_source","id":"hidden"}),prefetch=True)
        self.assertEqual(r["status"],"tool_budget_exhausted")
        self.assertEqual(r["unavailable_reads"],2)
        self.assertTrue(r["unchanged"])

    def test_zero_required(self):
        for mode in (False,True):
            e,t=fixture(CASES[4])
            r=run_arm(e,t,lambda _:candidate(t,[]),prefetch=mode)
            self.assertEqual((r["status"],r["host_reads"]),("valid",0))

    def test_paired_aggregate(self):
        def exchange(messages):
            ctx=json.loads(messages[1]["content"])
            return encode({"schema":"npr-gm-sandbox/1","turn_id":ctx["turn_id"],"revision":ctx["revision"],
                           "outcome_id":"o1","facts_used":ctx["authorized_outcomes"][0]["required_sources"],
                           "scene_draft":"Черновик."})
        r=experiment(exchange)
        self.assertEqual(len(r["rows"]),12)
        self.assertEqual(r["summary"]["direct"]["valid"],1)
        self.assertEqual(r["summary"]["prefetch"]["valid"],6)
        self.assertEqual(r["pairs"],{"prefetch_only":5,"both":1})
        self.assertTrue(r["all_unchanged"])
        self.assertFalse(r["training"])
