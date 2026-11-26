"""Local-only original engine smoke on a brand-new synthetic Git repository."""
import argparse
import copy
import json
import subprocess
import tempfile
from pathlib import Path
from .npr_bridge import git, pin, extract_engine, check, blob_sha, BridgeError

def synthetic_turn(n,prior=None):
    return {"schema":"npr-turn/2","id":f"s01.text.{n:03}","at":"0010-01-01T10:00:00",
            "place":"Учебная комната","player":"Осматриваю стол.","gm":"На столе лежит лист.",
            "line":"Осмотр стола.","review":{"predecessor":prior},"d":{"scene":{"present":[]}}}

def synthetic_packet(raw):
    return {"schema":"npr-next/1","cursor":"s01.text.001","next_id":"s01.text.002",
            "turn_file":blob_sha(raw),"lag":{"at":"0010-01-01T10:00:00","recorded_turn":"s01.text.001","dice_for":"s01.text.002"},
            "dice":[5,18],"fold_errors":[],"state_errors":[]}

def commit(root):
    git(root,"-c","core.autocrlf=false","add",".")
    git(root,"-c","user.name=Synthetic","-c","user.email=synthetic@example.invalid",
        "-c","core.hooksPath="+str(root/".no-hooks"),"commit","--no-gpg-sign","-qm","synthetic fixture")
    return git(root,"rev-parse","HEAD").decode().strip()

def run(root,revision):
    root=Path(root).resolve()
    pin(root,revision)
    original_tree=git(root,"rev-parse",revision+":rpg_engine").strip()
    with tempfile.TemporaryDirectory(prefix="npr-gm-original-smoke-") as temp:
        isolated=Path(temp)
        git(isolated,"init","-q","--template=")
        extract_engine(root,revision,isolated)
        code=commit(isolated)
        if git(isolated,"rev-parse",code+":rpg_engine").strip()!=original_tree:
            raise BridgeError("original_engine_tree_mismatch")
        recovery=isolated/"recovery";recovery.mkdir()
        raw=json.dumps(synthetic_turn(1),ensure_ascii=False).encode("utf-8")
        (recovery/"TURN.json").write_bytes(raw)
        (recovery/"NEXT.json").write_text(json.dumps(synthetic_packet(raw)),encoding="utf-8")
        state=commit(isolated)
        draft=synthetic_turn(2,"s01.text.001")
        stale=copy.deepcopy(draft);stale["review"]["predecessor"]="s01.text.000"
        no_scene=copy.deepcopy(draft);no_scene["d"]={}
        wrong_die=copy.deepcopy(draft);wrong_die["d"]["rolls"]=[{"value":18}]
        ooc=copy.deepcopy(draft);ooc["kind"]="ooc";ooc["at"]="0010-01-01T10:01:00"
        cases=(("clean",draft,True),("stale",stale,False),("missing_scene",no_scene,False),
               ("wrong_die",wrong_die,False),("ooc_clock",ooc,False))
        results=[]
        for name,candidate,expected in cases:
            r=check(isolated,code,state,candidate)
            results.append({"case":name,"expected":expected,"actual":r["mechanical_ok"],
                            "pass":r["status"]=="original_preflight" and r["mechanical_ok"]==expected and r["committed"] is False})
        unchanged=git(isolated,"rev-parse","HEAD").decode().strip()==state and not git(isolated,"status","--porcelain").strip()
        return {"schema":"original-preflight-smoke/1","original_tree_verified":True,
                "synthetic_state_only":True,"cases":results,"unchanged":unchanged,
                "passed":sum(x["pass"] for x in results),"total":len(results),
                "semantic_review":False,"committed":False}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",required=True,type=Path)
    p.add_argument("--code-rev",required=True)
    p.add_argument("--isolated-copy",required=True,action="store_true")
    a=p.parse_args()
    try: result=run(a.root,a.code_rev)
    except (BridgeError,ValueError,TypeError,KeyError,OSError,subprocess.TimeoutExpired):
        print(json.dumps({"status":"original_smoke_blocked","committed":False}))
        raise SystemExit(2)
    print(json.dumps(result))
    if not result["unchanged"] or result["passed"]!=result["total"]:raise SystemExit(1)

if __name__=="__main__":main()
