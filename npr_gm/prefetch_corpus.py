"""Draft rendering for mandatory-host-prefetch. Not an SFT exporter."""
import argparse
import json
from pathlib import Path
from .sandbox import Sandbox, Source, Outcome, ContractError, encode
from .prefetch_context import initial_messages
from .structured import load_drafts, validate_drafts
from .evaluate import load_cases

LEDGER=Path(__file__).resolve().parents[1]/"data"/"corpus_editorial_review_v3.jsonl"

def replay_prefetch(row):
    s=row["spec"]
    engine=Sandbox(resources=s["resources"],clock=s["clock"],
        sources=[Source(**x) for x in s["sources"]],
        outcomes=[Outcome(x["id"],x["cost"],x["minutes"],tuple(x["required_sources"]),x["public_text"]) for x in s["outcomes"]])
    # Current experimental scope has one authorized outcome only.
    if len(s["authorized_outcomes"])!=1:
        raise ContractError("prefetch_multi_outcome_unvalidated")
    t=engine.begin(s["player"],s["authorized_outcomes"],authorized_sources=s["authorized_sources"])
    reads=s["reads"]
    if len(reads)>4 or len(set(reads))!=len(reads) or not set(reads)<=set(t.source_ids):
        raise ContractError("invalid_draft_reads")
    context,required,prefetched,messages=initial_messages(engine,t,prefetch=True)
    if not set(required)<=set(reads):raise ContractError("missing_draft_citation")
    # Mandatory facts are host messages, never fabricated assistant calls.
    optional=[i for i in reads if i not in required]
    for i in optional:
        raw=encode({"tool":"read_source","id":i})
        result=engine.request(t,raw)
        if result.get("status")!="found":raise ContractError("optional_source_unavailable")
        messages.extend([{"role":"assistant","content":raw},
            {"role":"user","content":encode({"tool_result":result,"instruction":"Продолжи по контракту."})}])
    candidate=encode({"schema":"npr-gm-sandbox/1","turn_id":t.id,"revision":t.revision,
        "outcome_id":s["outcome_id"],"facts_used":reads,"scene_draft":s["scene_draft"]})
    engine.preview(t,candidate)
    if engine.revision!=0 or engine.resources!=s["resources"] or engine.clock!=s["clock"] or engine.receipt(t.id) is not None:
        raise ContractError("prefetch_replay_mutated")
    return messages+[{"role":"assistant","content":candidate}]

def eligible_drafts(rows,ledger):
    """Editorial candidates only, not approved for training."""
    index={}
    for review in ledger:
        if review["format"]!="structured":continue
        if review["id"] in index:raise ValueError("Duplicate editorial ID")
        index[review["id"]]=review
    if len(index)!=len(rows) or set(index)!={r["id"] for r in rows}:raise ValueError("Editorial coverage mismatch")
    out=[]
    for r in rows:
        v=index[r["id"]]
        if v["group_id"]!=r["group_id"] or v["train_eligible"] is not False or r["train_eligible"] is not False:
            raise ValueError("Editorial group or status mismatch")
        if v["decision"] in ("candidate","revised_candidate"):out.append(r)
        elif not v["decision"].startswith("hold_"):raise ValueError("Unknown editorial decision")
    return out

def load_ledger():
    return [json.loads(x) for x in LEDGER.read_text(encoding="utf-8").splitlines() if x.strip()]

def check_drafts():
    rows=load_drafts()
    validate_drafts(rows,load_cases())
    eligible=eligible_drafts(rows,load_ledger())
    messages=[replay_prefetch(r) for r in eligible]
    return {"drafts":len(rows),"editorial_candidates":len(eligible),"held":len(rows)-len(eligible),
            "assistant_targets":sum(m["role"]=="assistant" for x in messages for m in x),
            "optional_tool_targets":sum(m["role"]=="assistant" and "tool" in json.loads(m["content"]) for x in messages for m in x),
            "approved":0,"training":False}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--check-drafts",required=True,action="store_true")
    p.parse_args()
    print(json.dumps(check_drafts(),ensure_ascii=False))

if __name__=="__main__":main()
