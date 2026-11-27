"""Prefetch-only optional-retrieval draft groups. Never exported for training."""
import argparse
import json
from pathlib import Path
from .structured import load_drafts, validate_drafts
from .prefetch_corpus import replay_prefetch
from .evaluate import load_cases
from .corpus import normalized

ROOT=Path(__file__).resolve().parents[1]/"data"
DATA=ROOT/"prefetch_optional_draft.jsonl"
LEDGER=ROOT/"prefetch_optional_review.jsonl"

def load_optional():
    return load_drafts(DATA)

def check_optional(rows=None,ledger=None):
    rows=load_optional() if rows is None else rows
    ledger=load_drafts(LEDGER) if ledger is None else ledger
    validate_drafts(rows,load_cases())
    old=load_drafts()
    if {r["group_id"] for r in rows}&{r["group_id"] for r in old}:raise ValueError("Group collision")
    if {r["id"] for r in rows}&{r["id"] for r in old}:raise ValueError("ID collision")
    blocked={normalized(r["spec"][k]) for r in old for k in ("player","scene_draft")}
    if any(normalized(r["spec"][k]) in blocked for r in rows for k in ("player","scene_draft")):raise ValueError("Exact legacy overlap")
    reviews={v["id"]:v for v in ledger}
    if len(reviews)!=len(ledger) or set(reviews)!={r["id"] for r in rows}:raise ValueError("Review coverage mismatch")
    calls=targets=zero=0
    for r in rows:
        v=reviews[r["id"]]
        if r.get("contract_mode")!="host_prefetch_only" or v["group_id"]!=r["group_id"] or v.get("train_eligible") is not False:
            raise ValueError("Invalid draft route")
        if v["decision"]!="candidate":raise ValueError("Not editorial candidate")
        m=replay_prefetch(r)
        a=[json.loads(x["content"]) for x in m if x["role"]=="assistant"]
        c=sum("tool" in x for x in a)
        calls+=c;targets+=len(a);zero+=c==0
        # Distractor is authorized but not read; do not make all allowed IDs mandatory.
        if len(set(r["spec"]["authorized_sources"])-set(r["spec"]["reads"]))<1:
            raise ValueError("Missing unread distractor")
    return {"drafts":len(rows),"groups":len(rows),"assistant_targets":targets,"optional_tool_targets":calls,"zero_optional_calls":zero,"approved":0,"training":False}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--check-drafts",required=True,action="store_true")
    p.parse_args()
    print(json.dumps(check_optional(),ensure_ascii=False))

if __name__=="__main__":main()
