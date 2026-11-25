"""Local, read-only archive extraction and fail-closed SFT export.

No repository writes, no network, no inference. Candidate extraction is not approval.
Private files are written exclusively under a private_data directory.
"""
import argparse
import hashlib
import json
import re
import subprocess
import unicodedata
from pathlib import Path
from .evaluate import load_cases

HEX = re.compile(r"[a-f0-9]{40}")
HARD = {"C01","C02","C03","C04","C10"}
SOFT = {"C05","C06","C07","C08","C09","C10","C11"}

def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def canonical(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

def read_rows(raw):
    return [json.loads(line) for line in raw.splitlines() if line.strip()]

def unique_rows(rows):
    by_id = {}
    for row in rows:
        ident = row.get("id")
        if not isinstance(ident,str) or not ident.strip() or ident in by_id:
            raise ValueError("Missing or duplicate source turn ID")
        by_id[ident]=row
    return by_id

def extract_session(text_rows, delta_rows, campaign, revision, session, path, delta_path, history=3):
    if not HEX.fullmatch(revision): raise ValueError("Pin full commit SHA")
    if not re.fullmatch(r"[a-zA-Z0-9_-]+",campaign): raise ValueError("Use an opaque campaign ID")
    if not 0<=history<=10: raise ValueError("history out of range")
    unique_rows(text_rows)
    deltas=unique_rows(delta_rows)
    texts={row["id"] for row in text_rows}
    if set(deltas)-texts: raise ValueError("Unpaired delta records require explicit review")
    candidates=[]
    previous=[]
    for row in text_rows:
        ident=row["id"]
        for key in ("player","gm"):
            if not isinstance(row.get(key),str) or not row[key].strip(): raise ValueError("Missing original dialogue")
        source={"campaign":campaign,"commit":revision,"path":path,"turn_id":ident,
                "record_sha256":digest(canonical(row))}
        d=deltas.get(ident)
        delta_info=None if d is None else {"path":delta_path,"record_sha256":digest(canonical(d)),
                                           "has_object":isinstance(d.get("d"),dict)}
        candidates.append({
            "id":"A-"+digest(campaign+"\0"+path+"\0"+ident)[:24],
            "source":source,"provisional_group":campaign+":session:"+str(session),
            "group_id":None,"split":None,"rights_status":"pending_review",
            "review_status":"pending","train_eligible":False,
            "kind":row.get("kind") or "unspecified",
            "player":row["player"],
            "archive_response_for_review":row["gm"],
            "prior_records_for_review":previous[-history:] if history else [],
            "delta_after_for_review":d,
            "delta_source":delta_info,
            "context_snapshot":None,"chosen":None
        })
        previous.append({"id":ident,"player":row["player"],"gm":row["gm"],"kind":row.get("kind")})
    return candidates

def git(root,*args):
    return subprocess.run(["git","-C",str(root),*args],check=True,capture_output=True).stdout.decode("utf-8")

def pinned_extract(root,revision,campaign,sessions=None):
    if not HEX.fullmatch(revision): raise ValueError("Pin full commit SHA")
    actual=git(root,"rev-parse","--verify",revision+"^{commit}").strip()
    if actual!=revision: raise ValueError("Revision mismatch")
    paths=git(root,"ls-tree","-r","--name-only",revision).splitlines()
    allowed={p for p in paths if re.fullmatch(r"archive/session-\d{3}-turns\.jsonl",p)}
    if sessions is not None:
        wanted={f"archive/session-{s:03d}-turns.jsonl" for s in sessions}
        if not wanted<=allowed: raise ValueError("Requested session absent at pinned revision")
        allowed=wanted
    if not allowed: raise ValueError("No supported turn archive at revision")
    out=[]
    for path in sorted(allowed):
        session=int(re.search(r"session-(\d{3})",path).group(1))
        dp=f"archive/deltas/session-{session:03d}.jsonl"
        rows=read_rows(git(root,"show",revision+":"+path))
        delta=read_rows(git(root,"show",revision+":"+dp)) if dp in paths else []
        out.extend(extract_session(rows,delta,campaign,revision,session,path,dp))
    if len({r["id"] for r in out})!=len(out): raise ValueError("Duplicate candidate ID")
    return out

def private_output(path, source=None):
    p=Path(path).resolve()
    if "private_data" not in p.parts[:-1]: raise ValueError("Output must be within private_data")
    if source is not None and p.is_relative_to(Path(source).resolve()):
        raise ValueError("Do not write into source repository")
    p.parent.mkdir(parents=True,exist_ok=True)
    return p

def write_jsonl(path,rows):
    with path.open("x",encoding="utf-8") as f:
        for row in rows: f.write(json.dumps(row,ensure_ascii=False)+"\n")

def normalized(text):
    return " ".join(unicodedata.normalize("NFKC",text).casefold().split())

def reference_strings(cases):
    blocked=set()
    for c in cases:
        blocked.add(normalized(c["player"]))
        for k in ("chosen","rejected"): blocked.add(normalized(c[k]["scene"]))
    return blocked

def review_export(rows, split_map, smoke_cases):
    """Strict export. Expert review and semantic leakage review remain human tasks."""
    blocked=reference_strings(smoke_cases)
    groups={}
    seen=set()
    source_seen=set()
    export=[]
    for row in rows:
        if row.get("review_status")!="approved" or row.get("train_eligible") is not True:
            continue
        if row.get("rights_status")!="approved_for_training": raise ValueError("Rights not approved")
        if row.get("semantic_leakage_review")!="pass": raise ValueError("Semantic leakage review missing")
        if row.get("snapshot_review")!="verified_as_of_turn": raise ValueError("Unverified context")
        if not isinstance(row.get("reviewer_id"),str) or not row["reviewer_id"].strip(): raise ValueError("Missing reviewer")
        g=row.get("group_id")
        s=row.get("split")
        if not isinstance(g,str) or not g.strip() or s not in {"train","dev","holdout"} or split_map.get(g)!=s:
            raise ValueError("Explicit group split missing or inconsistent")
        if g in groups and groups[g]!=s: raise ValueError("Cross-split group")
        groups[g]=s
        source=row.get("source")
        if not isinstance(source,dict) or not source: raise ValueError("Missing source")
        sk=canonical(source)
        if sk in source_seen: raise ValueError("Duplicate source record")
        source_seen.add(sk)
        app=row.get("applicable_ids")
        if not isinstance(app,list) or not set(app)<=HARD|SOFT or not {"C01","C02","C03","C04"}<=set(app):
            raise ValueError("Missing mandatory competence review")
        hard=row.get("hard_results",{})
        if any(hard.get(k)!="pass" for k in set(app)&HARD): raise ValueError("Hard gate failed or unknown")
        scores=row.get("soft_scores",{})
        if any(type(scores.get(k)) is not int or not 0<=scores[k]<=4 for k in set(app)&SOFT):
            raise ValueError("Missing applicable soft score")
        key=row.get("key_competence")
        if key not in set(app)&SOFT or scores[key]<3: raise ValueError("Key competence below threshold")
        prompt=row.get("context_snapshot")
        player=row.get("player")
        chosen=row.get("chosen")
        if not all(isinstance(x,str) and x.strip() for x in (prompt,player,chosen)): raise ValueError("Missing reviewed dialogue")
        if normalized(player) in blocked or normalized(chosen) in blocked: raise ValueError("Exact public eval overlap")
        pair=(normalized(prompt),normalized(player),normalized(chosen))
        if pair in seen: raise ValueError("Duplicate dialogue")
        seen.add(pair)
        if s=="train":
            export.append({"messages":[{"role":"system","content":"Ты ведущий текстовой RPG. Соблюдай подтверждённое состояние мира, субъектность игрока и разрешённую механику."},
                                       {"role":"user","content":prompt+"\n\nЗаявка игрока:\n"+player},
                                       {"role":"assistant","content":chosen}]})
    if not export: raise ValueError("No approved train records")
    return export

def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="cmd",required=True)
    ex=sub.add_parser("extract");ex.add_argument("--root",required=True,type=Path);ex.add_argument("--revision",required=True);ex.add_argument("--campaign",required=True);ex.add_argument("--session",type=int,nargs="+");ex.add_argument("--output",required=True,type=Path)
    ep=sub.add_parser("export-sft");ep.add_argument("--reviewed",required=True,type=Path);ep.add_argument("--split-map",required=True,type=Path);ep.add_argument("--output",required=True,type=Path)
    a=p.parse_args()
    if a.cmd=="extract":
        out=private_output(a.output,a.root)
        rows=pinned_extract(a.root,a.revision,a.campaign,a.session)
        write_jsonl(out,rows)
        print(json.dumps({"candidates":len(rows),"object_delta":sum(bool((r["delta_source"] or {}).get("has_object")) for r in rows),"ooc":sum(r["kind"]=="ooc" for r in rows),"approved":0}))
    else:
        out=private_output(a.output)
        rows=read_rows(a.reviewed.read_text(encoding="utf-8"))
        mapping=json.loads(a.split_map.read_text(encoding="utf-8"))
        approved=review_export(rows,mapping,load_cases())
        write_jsonl(out,approved)
        print(json.dumps({"exported_reviewed_train_records":len(approved)}))
if __name__=="__main__":main()
