"""CPU-only tokenizer preflight, no model weights, training or private inputs."""
import hashlib
import importlib.metadata
import json
import statistics
from pathlib import Path
from .structured import load_drafts,replay
from .prefetch_corpus import replay_prefetch,eligible_drafts,load_ledger
from .optional_corpus import load_optional,check_optional
from .train_qlora import encode_messages,encode_targets,smoke_messages

MODEL="Qwen/Qwen3.5-4B"
REVISION="bbaaae1f57d79c7bf696d619ffbbb0c2c8e5ef98"
FILES=["tokenizer.json","tokenizer_config.json","chat_template.jinja","config.json",
       "merges.txt","vocab.json","special_tokens_map.json","added_tokens.json"]

def main():
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    directory=Path(snapshot_download(MODEL,revision=REVISION,allow_patterns=FILES,token=False))
    tok=AutoTokenizer.from_pretrained(directory,local_files_only=True,trust_remote_code=False)
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.iterdir() if p.is_file() and p.name in FILES}
    rows=[replay(x) for x in load_drafts()]
    legacy_pass=0;legacy_fail=0;lengths=[];target_lengths=[];failures=[]
    for row in rows:
        try:
            encode_messages(tok,row,2048)
            legacy_pass+=1
        except ValueError:
            legacy_fail+=1
        try:
            samples=encode_targets(tok,row,2048)
            expected=[m["content"].strip() for m in row if m["role"]=="assistant"]
            if len(samples)!=len(expected):raise ValueError("target_count")
            for sample,content in zip(samples,expected):
                ids=[x for x in sample["labels"] if x!=-100]
                decoded=tok.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)
                # Only the assistant continuation and its exact terminator may be targets.
                if decoded.rstrip("\n")!=content+"<|im_end|>":raise ValueError("target_decode_mismatch")
                lengths.append(len(sample["input_ids"]));target_lengths.append(len(ids))
        except (ValueError,TypeError) as e:
            failures.append(type(e).__name__+":"+str(e))
    prefetch_rows=[replay_prefetch(r) for r in eligible_drafts(load_drafts(),load_ledger())]
    pf_lengths=[];pf_target_lengths=[];pf_failures=[]
    for row in prefetch_rows:
        try:
            samples=encode_targets(tok,row,2048)
            expected=[m["content"].strip() for m in row if m["role"]=="assistant"]
            if len(samples)!=len(expected):raise ValueError("target_count")
            for sample,content in zip(samples,expected):
                ids=[x for x in sample["labels"] if x!=-100]
                decoded=tok.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)
                if decoded.rstrip("\n")!=content+"<|im_end|>":raise ValueError("target_decode_mismatch")
                pf_lengths.append(len(sample["input_ids"]));pf_target_lengths.append(len(ids))
        except (ValueError,TypeError) as e:
            pf_failures.append(type(e).__name__+":"+str(e))
    def measure(dialogues):
        lengths=[];target_lengths=[];fail=[]
        for row in dialogues:
            try:
                samples=encode_targets(tok,row,2048)
                expected=[m["content"].strip() for m in row if m["role"]=="assistant"]
                if len(samples)!=len(expected):raise ValueError("target_count")
                for sample,content in zip(samples,expected):
                    ids=[x for x in sample["labels"] if x!=-100]
                    decoded=tok.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)
                    if decoded.rstrip("\n")!=content+"<|im_end|>":raise ValueError("target_decode_mismatch")
                    lengths.append(len(sample["input_ids"]));target_lengths.append(len(ids))
            except (ValueError,TypeError) as e:
                fail.append(type(e).__name__+":"+str(e))
        return {"dialogues":len(dialogues),"targets":len(lengths),"input_min_median_max":[min(lengths),statistics.median(lengths),max(lengths)] if lengths else [],
                "target_lengths_min_max":[min(target_lengths),max(target_lengths)] if target_lengths else [],
                "over_512":sum(n>512 for n in lengths),"over_1024":sum(n>1024 for n in lengths),"failures":fail}
    check_optional()
    optional=measure([replay_prefetch(r) for r in load_optional()])
    literary_path=Path(__file__).resolve().parents[1]/"data"/"literary_train_draft.jsonl"
    literary=measure([[{"role":"system","content":"Ты ведущий текстовой RPG. Соблюдай подтверждённое состояние мира, субъектность игрока и разрешённую механику."},
        {"role":"user","content":r["context_snapshot"]+"\n\nЗаявка игрока:\n"+r["player"]},{"role":"assistant","content":r["chosen"]}] for r in load_drafts(literary_path)])
    smoke=encode_targets(tok,smoke_messages()[0]["messages"],512)
    result={"model":MODEL,"revision":REVISION,"packages":{k:importlib.metadata.version(k) for k in ("transformers","tokenizers","huggingface-hub","jinja2")},
        "file_sha256":hashes,"dialogues":len(rows),"legacy_full_mask_pass":legacy_pass,"legacy_full_mask_fail":legacy_fail,
        "separate_targets":len(lengths),"target_lengths_min_max":[min(target_lengths),max(target_lengths)] if target_lengths else [],
        "input_min_median_max":[min(lengths),statistics.median(lengths),max(lengths)] if lengths else [],
        "over_512":sum(n>512 for n in lengths),"over_1024":sum(n>1024 for n in lengths),
        "prefetch":{"dialogues":len(prefetch_rows),"targets":len(pf_lengths),"input_min_median_max":[min(pf_lengths),statistics.median(pf_lengths),max(pf_lengths)] if pf_lengths else [],"target_lengths_min_max":[min(pf_target_lengths),max(pf_target_lengths)] if pf_target_lengths else [],"over_512":sum(n>512 for n in pf_lengths),"failures":pf_failures},"optional_prefetch":optional,"literary":literary,"smoke_input_tokens":len(smoke[0]["input_ids"]),"failures":failures,"weights_loaded":False,"training":False}
    print(json.dumps(result,ensure_ascii=False),flush=True)
    if failures or len(lengths)!=32 or pf_failures or len(pf_lengths)!=15 or optional['failures'] or optional['targets']!=15 or literary['failures'] or literary['targets']!=6:raise SystemExit(1)
if __name__=="__main__":main()
