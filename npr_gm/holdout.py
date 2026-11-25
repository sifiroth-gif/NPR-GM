"""Local frozen holdout runner. No case text, criteria or outputs in CI logs."""
import argparse
import hashlib
import json
import re
from pathlib import Path
from .evaluate import check, generate, load_cases
from .corpus import private_output, normalized, reference_strings

def load_private(path, expected_hash):
    path=Path(path).resolve()
    if "private_data" not in path.parts[:-1]: raise ValueError("Holdout must remain under private_data")
    raw=path.read_bytes()
    if not re.fullmatch("[a-f0-9]{64}",expected_hash) or hashlib.sha256(raw).hexdigest()!=expected_hash:
        raise ValueError("Frozen holdout checksum mismatch")
    rows=[json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    if not rows or len({r["id"] for r in rows})!=len(rows): raise ValueError("Empty or duplicate holdout")
    blocked=reference_strings(load_cases())
    for r in rows:
        if r.get("split")!="holdout" or r.get("source")!="original_synthetic" or not r.get("group") or not r.get("manual_criteria"):
            raise ValueError("Invalid holdout provenance")
        if not all(isinstance(r[k],str) and r[k].strip() for k in ("context","player")):
            raise ValueError("Missing scenario")
        if normalized(r["player"]) in blocked: raise ValueError("Public benchmark exact overlap")
        for allowed,required in (("allowed_facts","required_facts"),("allowed_actions","required_actions")):
            if not isinstance(r[allowed],list) or not isinstance(r[required],list) or not set(r[required])<=set(r[allowed]):
                raise ValueError("Invalid allowed/required IDs")
        if type(r["clock_min"]) is not int or type(r["clock_max"]) is not int or not 0<=r["clock_min"]<=r["clock_max"]:
            raise ValueError("Invalid clock bounds")
        if not isinstance(r["forbidden_text"],list): raise ValueError("Invalid text checks")
    return rows

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",type=Path,required=True);p.add_argument("--sha256",required=True)
    p.add_argument("--check-only",action="store_true");p.add_argument("--model")
    p.add_argument("--endpoint",default="http://127.0.0.1:8080/v1/chat/completions")
    p.add_argument("--output",type=Path);p.add_argument("--timeout",type=int,default=180)
    p.add_argument("--max-tokens",type=int,default=384)
    a=p.parse_args();rows=load_private(a.data,a.sha256)
    if a.check_only:
        print(json.dumps({"cases":len(rows),"groups":len({r["group"] for r in rows}),"sha256":a.sha256,"status":"frozen_unrun"}))
        return
    if not a.model or not a.output or not 1<=a.timeout<=300 or not 1<=a.max_tokens<=2048: p.error("Missing/bad model, output or limits")
    out=private_output(a.output)
    passed=responses=0
    with out.open("x",encoding="utf-8") as f:
        for r in rows:
            try:
                answer,raw,seconds,usage=generate(r,a.endpoint,a.model,a.timeout,max_tokens=a.max_tokens,thinking=False)
                errors=check(r,answer)
                responses+=1
            except Exception as e:
                answer,raw,seconds,usage=None,None,None,{}
                errors=["transport:"+type(e).__name__]
            passed+=not errors
            f.write(json.dumps({"id":r["id"],"group":r["group"],"model":a.model,"dataset_sha256":a.sha256,"answer":answer,"raw":raw,"seconds":seconds,"usage":usage,"errors":errors,"manual_review":None},ensure_ascii=False)+"\n");f.flush()
    print(json.dumps({"attempted":len(rows),"responses":responses,"automatic_pass":passed,"manual_review":"not_performed"}))
    if responses!=len(rows): raise SystemExit(2)
if __name__=="__main__":main()
