"""Read-only, pinned original NPR preflight bridge.

Only run against a deliberately isolated local copy of a trusted source repository.
No GitHub API, TURN writes, fold, publication, or model-facing raw state.
"""
import argparse
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

SHA=re.compile(r"[0-9a-f]{40}")
NEXT="recovery/NEXT.json"
TURN="recovery/TURN.json"

class BridgeError(ValueError):
    pass

def git(root,*args):
    p=subprocess.run(["git","-C",str(root),*args],capture_output=True,check=False,timeout=45)
    if p.returncode:raise BridgeError("local_git_command_failed")
    return p.stdout

def pin(root,rev):
    if not isinstance(rev,str) or not SHA.fullmatch(rev):
        raise BridgeError("full_commit_required")
    if git(root,"rev-parse","--verify",rev+"^{commit}").decode().strip()!=rev:
        raise BridgeError("revision_mismatch")
    return rev

def blob_sha(raw):
    return hashlib.sha1(b"blob "+str(len(raw)).encode()+b"\0"+raw).hexdigest()

def read_object(root,rev,path):
    raw=git(root,"show",rev+":"+path)
    expected=git(root,"rev-parse",rev+":"+path).decode().strip()
    if blob_sha(raw)!=expected:raise BridgeError("blob_mismatch")
    if len(raw)>8_000_000:raise BridgeError("object_too_large")
    try:
        obj=json.loads(raw)
    except ValueError as e:raise BridgeError("invalid_json") from e
    if not isinstance(obj,dict):raise BridgeError("expected_object")
    return obj,expected

def ancestry(root,old,new):
    git(root,"merge-base","--is-ancestor",old,new)

def inspect_anchor(packet,actual,actual_sha,receipts):
    """Cold-start boundary: don't trust NEXT alone or fabricate receipts."""
    if packet.get("schema")!="npr-next/1" or actual.get("schema")!="npr-turn/2":
        raise BridgeError("unsupported_original_schema")
    if packet.get("fold_errors") or packet.get("state_errors") or (packet.get("lag_status") or {}).get("write_allowed") is False:
        raise BridgeError("fold_or_state_blocked")
    cursor=packet.get("cursor");next_id=packet.get("next_id");base_sha=packet.get("turn_file")
    if not all(isinstance(x,str) and x for x in (cursor,next_id,base_sha)) or not SHA.fullmatch(base_sha):
        raise BridgeError("invalid_anchor")
    recorded=(packet.get("lag") or {}).get("recorded_turn",cursor)
    if not receipts:
        if actual.get("id")!=cursor or recorded!=cursor or base_sha!=actual_sha:
            raise BridgeError("stale_or_changed_turn_requires_receipts")
        return {"mode":"fresh","expected_sha":actual_sha}
    prior=cursor
    for i,rec in enumerate(receipts):
        t=rec["turn"]
        if t.get("schema")!="npr-turn/2" or (t.get("review") or {}).get("predecessor")!=prior:
            raise BridgeError("receipt_predecessor_mismatch")
        if i==0 and t.get("id")!=next_id:raise BridgeError("first_receipt_not_next")
        if t.get("id")==prior or t.get("amend"):raise BridgeError("amend_or_replay_requires_refresh")
        prior=t.get("id")
    if receipts[-1]["turn"]!=actual or receipts[-1]["sha"]!=actual_sha:
        raise BridgeError("receipt_chain_not_latest")
    if recorded!=cursor and not any(x["turn"].get("id")==recorded and x["sha"]==base_sha for x in receipts):
        raise BridgeError("recorded_anchor_receipt_missing")
    return {"mode":"receipt_projection","expected_sha":actual_sha}

def read_snapshot(root,state_rev,receipt_commits=()):
    pin(root,state_rev)
    packet,_=read_object(root,state_rev,NEXT)
    actual,actual_sha=read_object(root,state_rev,TURN)
    if len(receipt_commits)>32:raise BridgeError("receipt_budget")
    receipts=[];prev=None;seen=set()
    for commit in receipt_commits:
        pin(root,commit);ancestry(root,commit,state_rev)
        if prev is not None:ancestry(root,prev,commit)
        t,sha=read_object(root,commit,TURN)
        if sha in seen:raise BridgeError("duplicate_receipt_blob")
        seen.add(sha);prev=commit
        receipts.append({"turn":t,"sha":sha})
    anchor=inspect_anchor(packet,actual,actual_sha,receipts)
    # When NEXT names its folded predecessor, prove that the first receipt's
    # parent actually contains that exact predecessor blob. A mere ID is not proof.
    if receipts and (packet.get("lag") or {}).get("recorded_turn",packet["cursor"])==packet["cursor"]:
        parent=git(root,"rev-parse",receipt_commits[0]+"^1").decode().strip()
        old,oldsha=read_object(root,parent,TURN)
        if old.get("id")!=packet["cursor"] or oldsha!=packet["turn_file"]:
            raise BridgeError("receipt_base_blob_mismatch")
    return packet,receipts,anchor

def extract_engine(root,code_rev,dest):
    pin(root,code_rev)
    raw=git(root,"archive","--format=tar",code_rev,"rpg_engine")
    if len(raw)>30_000_000:raise BridgeError("engine_archive_budget")
    with tarfile.open(fileobj=io.BytesIO(raw),mode="r:") as tf:
        for member in tf.getmembers():
            p=Path(member.name)
            if p.is_absolute() or ".." in p.parts or not p.parts or p.parts[0]!="rpg_engine":
                raise BridgeError("unsafe_engine_path")
            if member.isdir():continue
            if not member.isfile():raise BridgeError("unsafe_engine_member")
            target=dest/p
            target.parent.mkdir(parents=True,exist_ok=True)
            data=tf.extractfile(member).read()
            target.write_bytes(data)
    if not (dest/"rpg_engine"/"preflight.py").is_file():raise BridgeError("original_preflight_missing")

WORKER = """
import json,sys
sys.path.insert(0,sys.argv[1])
from rpg_engine import overlay,preflight
p=json.load(sys.stdin)
projected=overlay.project(p['packet'],p['receipts'])
if (projected.get('lag_status') or {}).get('write_allowed') is False:
    print(json.dumps({'status':'projection_blocked','mechanical_ok':False}))
    raise SystemExit(0)
if projected.get('cursor')!=p['expected_cursor'] or projected.get('turn_file')!=p['expected_sha']:
    print(json.dumps({'status':'projection_anchor_mismatch','mechanical_ok':False}))
    raise SystemExit(0)
result=preflight.inspect(p['packet'],p['draft'],receipts=p['receipts'],used=p['used'])
print(json.dumps({'status':'original_preflight','mechanical_ok':result.get('mechanical_ok') is True,
                  'error_count':len(result.get('errors') or []),'review_count':len(result.get('review') or []),
                  'checklist_count':len(result.get('checklist') or []),
                  'semantic_review_required':True,'committed':False}))
"""

def check(root,code_rev,state_rev,draft,*,receipt_commits=(),used=()):
    root=Path(root).resolve()
    pin(root,code_rev)
    if not isinstance(draft,dict) or draft.get("schema")!="npr-turn/2":
        raise BridgeError("original_turn_required")
    if not all(isinstance(x,str) and x for x in used):raise BridgeError("invalid_used_ids")
    packet,receipts,anchor=read_snapshot(root,state_rev,receipt_commits)
    expected_cursor=receipts[-1]["turn"]["id"] if receipts else packet["cursor"]
    payload={"packet":packet,"receipts":receipts,"draft":draft,"used":list(used),
             "expected_cursor":expected_cursor,"expected_sha":anchor["expected_sha"]}
    with tempfile.TemporaryDirectory(prefix="npr-gm-preflight-") as tmp:
        dest=Path(tmp);extract_engine(root,code_rev,dest)
        # Trusted original code runs isolated from checkout and credentials.
        # This is not a security sandbox for hostile Python.
        env={k:os.environ[k] for k in ("PATH","SYSTEMROOT","WINDIR","LANG","LC_ALL") if k in os.environ}
        proc=subprocess.run([sys.executable,"-I","-B","-c",WORKER,str(dest)],
            input=json.dumps(payload,ensure_ascii=True),encoding="utf-8",capture_output=True,
            cwd=dest,env=env,timeout=60)
        if proc.returncode:raise BridgeError("original_preflight_execution_failed")
        try:result=json.loads(proc.stdout)
        except ValueError as e:raise BridgeError("invalid_preflight_response") from e
    if not isinstance(result,dict) or type(result.get("mechanical_ok")) is not bool:
        raise BridgeError("invalid_preflight_response")
    result["snapshot_mode"]=anchor["mode"]
    result["committed"]=False
    result["semantic_review_required"]=True
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--code-rev",required=True);p.add_argument("--state-rev",required=True)
    p.add_argument("--draft",type=Path,required=True)
    p.add_argument("--receipt-commit",action="append",default=[])
    p.add_argument("--used",action="append",default=[])
    p.add_argument("--isolated-copy",action="store_true",required=True,
                   help="Confirm root is a deliberately isolated local copy")
    a=p.parse_args()
    if "private_data" not in a.draft.resolve().parts[:-1]:p.error("Draft must remain under private_data")
    try:
        result=check(a.root,a.code_rev,a.state_rev,json.loads(a.draft.read_text(encoding="utf-8")),
                     receipt_commits=a.receipt_commit,used=a.used)
    except (BridgeError,ValueError,TypeError,KeyError,OSError,subprocess.TimeoutExpired):
        print(json.dumps({"status":"bridge_blocked","mechanical_ok":False,"committed":False}))
        raise SystemExit(2)
    print(json.dumps(result))
    if not result["mechanical_ok"]:raise SystemExit(1)
if __name__=="__main__":main()
