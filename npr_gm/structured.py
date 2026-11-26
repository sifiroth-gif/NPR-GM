"""Replay original draft conversations against sandbox; separate guarded SFT export."""
import argparse
import json
from pathlib import Path
from .sandbox import Sandbox,Source,Outcome,encode,ContractError
from .transport import propose
from .corpus import normalized,reference_strings,canonical,private_output,write_jsonl
from .evaluate import load_cases

DATA=Path(__file__).resolve().parents[1]/"data"/"structured_train_draft.jsonl"

def load_drafts(path=DATA):
    return [json.loads(s) for s in Path(path).read_text(encoding="utf-8").splitlines() if s.strip()]

def replay(row):
    s=row["spec"]
    engine=Sandbox(resources=s["resources"],clock=s["clock"],
        sources=[Source(**x) for x in s["sources"]],
        outcomes=[Outcome(x["id"],x["cost"],x["minutes"],tuple(x["required_sources"]),x["public_text"]) for x in s["outcomes"]])
    t=engine.begin(s["player"],s["authorized_outcomes"],authorized_sources=s["authorized_sources"])
    reads=s["reads"]
    if len(reads)>4 or len(set(reads))!=len(reads) or not set(reads)<=set(t.source_ids):
        raise ContractError("invalid_draft_reads")
    candidate=encode({"schema":"npr-gm-sandbox/1","turn_id":t.id,"revision":t.revision,
        "outcome_id":s["outcome_id"],"facts_used":reads,"scene_draft":s["scene_draft"]})
    replies=iter([encode({"tool":"read_source","id":k}) for k in reads]+[candidate])
    captured=[]
    def exchange(messages):
        captured[:] = messages
        return next(replies)
    result=propose(engine,t,exchange)
    if result["tool_calls"]!=len(reads) or engine.revision!=0:
        raise ContractError("invalid_replay")
    return captured+[{"role":"assistant","content":result["candidate"]}]

def validate_drafts(rows, cases):
    blocked=reference_strings(cases)
    ids=set();groups=set()
    for r in rows:
        if r["id"] in ids or r["group_id"] in groups: raise ValueError("Duplicate draft/group")
        ids.add(r["id"]);groups.add(r["group_id"])
        if r["source"]["kind"]!="original_synthetic" or r["review_status"]!="pending" or r["train_eligible"] is not False or r["split"]!="unassigned":
            raise ValueError("Draft must not be approved")
        if any(normalized(r["spec"][k]) in blocked for k in ("player","scene_draft")):
            raise ValueError("Exact public eval overlap")
        replay(r)
    return {"drafts":len(rows),"groups":len(groups),"assistant_targets":sum(len(r["spec"]["reads"])+1 for r in rows),"approved":0}

def export_reviewed(rows,mapping,cases):
    blocked=reference_strings(cases)
    seen=set();sources=set();out=[]
    for r in rows:
        if r.get("review_status")!="approved" or r.get("train_eligible") is not True:
            continue
        if r.get("rights_status")!="approved_for_training" or r.get("semantic_leakage_review")!="pass" or r.get("snapshot_review")!="verified_as_of_turn":
            raise ValueError("Missing review gate")
        if not isinstance(r.get("reviewer_id"),str) or not r["reviewer_id"].strip():raise ValueError("Missing reviewer")
        g=r.get("group_id");split=r.get("split")
        if not isinstance(g,str) or not g or split not in {"train","dev","holdout"} or mapping.get(g)!=split:
            raise ValueError("Group split mismatch")
        if not isinstance(r.get("source"),dict) or not r["source"]:raise ValueError("Missing provenance")
        sk=canonical(r["source"])
        if sk in sources or r.get("id") in seen:raise ValueError("Duplicate reviewed source/ID")
        sources.add(sk);seen.add(r.get("id"))
        app=set(r.get("applicable_ids",[]));hard=r.get("hard_results",{});soft=r.get("soft_scores",{})
        if not {"C01","C02","C03","C04","C10"}<=app or not app<={"C"+str(i).zfill(2) for i in range(1,12)}:
            raise ValueError("Missing mandatory review")
        if any(hard.get(k)!="pass" for k in app&{"C01","C02","C03","C04","C10","C11"}):
            raise ValueError("Critical review failed")
        if any(type(soft.get(k)) is not int or not 0<=soft[k]<=4 for k in app&{"C05","C06","C07","C08","C09","C10","C11"}):
            raise ValueError("Missing soft review")
        if soft["C10"]<3:raise ValueError("Tool competency below threshold")
        if any(normalized(r["spec"][k]) in blocked for k in ("player","scene_draft")):
            raise ValueError("Exact public eval overlap")
        messages=replay(r)
        if split=="train":out.append({"messages":messages})
    if not out:raise ValueError("No approved structured train records")
    if len({canonical(x) for x in out})!=len(out):raise ValueError("Duplicate dialogue")
    return out

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--check-drafts",action="store_true")
    p.add_argument("--reviewed",type=Path);p.add_argument("--split-map",type=Path);p.add_argument("--output",type=Path)
    a=p.parse_args()
    if a.check_drafts:
        print(json.dumps(validate_drafts(load_drafts(),load_cases())))
        return
    if not a.reviewed or not a.split_map or not a.output:p.error("Use --check-drafts or reviewed/split-map/output")
    for path in (a.reviewed,a.split_map):
        if "private_data" not in path.resolve().parts[:-1]:raise ValueError("Private inputs required")
    rows=export_reviewed(load_drafts(a.reviewed),json.loads(a.split_map.read_text(encoding="utf-8")),load_cases())
    write_jsonl(private_output(a.output),rows)
    print(json.dumps({"exported":len(rows)}))
if __name__=="__main__":main()
