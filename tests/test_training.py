import unittest
from argparse import Namespace
from npr_gm.train_qlora import encode_messages,collate,validate_args,smoke_messages,local_input

class FakeTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return [1,2,3] if len(messages)==2 else [1,2,3,4,5]

class TrainTests(unittest.TestCase):
    def test_mask(self):
        x=encode_messages(FakeTokenizer(),smoke_messages()[0]["messages"],10)
        self.assertEqual(x["labels"],[-100,-100,-100,4,5])
    def test_overlong(self):
        with self.assertRaises(ValueError): encode_messages(FakeTokenizer(),smoke_messages()[0]["messages"],4)
    def test_prefix_mismatch(self):
        class Bad(FakeTokenizer):
            def apply_chat_template(self,messages,**kwargs): return [8] if len(messages)==2 else [1,2]
        with self.assertRaises(ValueError): encode_messages(Bad(),smoke_messages()[0]["messages"],10)
    def test_padding(self):
        r={"input_ids":[1,2],"attention_mask":[1,1],"labels":[-100,2]}
        s={"input_ids":[3],"attention_mask":[1],"labels":[3]}
        self.assertEqual(collate([r,s],0)["labels"],[[-100,2],[3,-100]])
    def args(self):
        return Namespace(max_length=512,steps=3,accum=1,smoke=True,reviewed=None,split_map=None,revision="a"*40,model="Qwen/Qwen3.5-4B",lr=0.0001)
    def test_bounds(self):
        a=self.args();validate_args(a);a.steps=4
        with self.assertRaises(ValueError): validate_args(a)
    def test_no_unreviewed_pilot(self):
        a=self.args();a.smoke=False
        with self.assertRaises(ValueError): validate_args(a)
    def test_pin(self):
        a=self.args();a.revision="main"
        with self.assertRaises(ValueError): validate_args(a)
    def test_local_input(self):
        with self.assertRaises(ValueError): local_input("data/synthetic_train_draft.jsonl")

    def test_multi_turn_mask(self):
        class Tokenizer:
            def apply_chat_template(self,messages,**kwargs):
                ids=[]
                for m in messages:
                    ids += [{"system":1,"user":2,"assistant":3}[m["role"]],int(m["content"]),9]
                if kwargs["add_generation_prompt"]:ids.append(3)
                return ids
        messages=[{"role":r,"content":str(i+10)} for i,r in enumerate(["system","user","assistant","user","assistant"])]
        x=encode_messages(Tokenizer(),messages,100)
        self.assertEqual(x["labels"],[-100]*7+[12,9]+[-100]*4+[14,9])
    def test_bad_role_order(self):
        with self.assertRaises(ValueError):encode_messages(FakeTokenizer(),[{"role":"system"},{"role":"assistant"},{"role":"user"}],100)
    def test_structured_smoke_mixing(self):
        a=self.args();a.structured=True
        with self.assertRaises(ValueError):validate_args(a)
