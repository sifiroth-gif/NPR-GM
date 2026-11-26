"""Human editorial lexical lookup; separately guarded model-facing retrieval."""
import argparse
import json
import re
from .knowledge import load, validate, DOMAINS

def tokens(text):
    return set(re.findall(r"[a-zа-я0-9]{3,}", text.casefold().replace("ё","е")))

def match(q, words):
    # Conservative prefix matching for common inflections; no semantic claims.
    return any(q==w or (min(len(q),len(w))>=5 and q[:5]==w[:5]) for w in words)

def retrieve(query, cards, domain=None, scope=None, limit=3, allow_draft=False):
    if domain is not None and domain not in DOMAINS: raise ValueError("Unknown domain")
    if not 1<=limit<=10: raise ValueError("limit out of range")
    terms=tokens(query)
    results=[]
    for c in cards:
        if domain and c["domain"]!=domain: continue
        if scope and c["scope"]!=scope: continue
        if c["status"]!="reviewed_reference" and not (allow_draft and c["status"]=="draft_reference"): continue
        title=tokens(c["topic"])
        body=tokens(c["fact"]+" "+c["application"])
        score=sum(3*match(t,title)+match(t,body) for t in terms)
        if score:
            result={k:c[k] for k in ("id","domain","topic","fact","application","source_id","url","scope","status","review","limitations","train_eligible")}
            result["lexical_score"]=score
            results.append(result)
    return sorted(results,key=lambda x:(-x["lexical_score"],x["id"]))[:limit]

def retrieve_for_model(query, registry, cards, domain=None, scope=None, limit=3):
    """Fail closed: neither public access nor a reviewed status implies AI permission.
    This is an interface gate, not evidence that any supplied permission is authentic.
    Only trusted host may supply reviewed local records; never model-generated flags.
    """
    sources={s["id"]:s for s in registry["sources"]}
    eligible=[]
    for c in cards:
        s=sources.get(c.get("source_id"),{})
        if c.get("status")!="reviewed_reference":continue
        if c.get("model_use")!="approved_for_model_retrieval" or s.get("model_use")!="approved_for_model_retrieval":continue
        if not all(isinstance(x,str) and x.strip() for x in (s.get("permission_ref"),c.get("permission_ref"),c.get("expert_reviewer_id"),c.get("review_date"))):continue
        eligible.append(c)
    return retrieve(query,eligible,domain,scope,limit,allow_draft=False)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("query")
    p.add_argument("--domain",choices=sorted(DOMAINS))
    p.add_argument("--scope")
    p.add_argument("--limit",type=int,default=3)
    p.add_argument("--allow-draft",action="store_true")
    p.add_argument("--model-facing",action="store_true",help="Require explicit permission and expert review; never returns drafts")
    a=p.parse_args()
    if a.model_facing and a.allow_draft:p.error("Drafts cannot be model-facing")
    registry,cards=load()
    if validate(registry,cards): raise SystemExit("Knowledge corpus integrity failure")
    results=retrieve_for_model(a.query,registry,cards,a.domain,a.scope,a.limit) if a.model_facing else retrieve(a.query,cards,a.domain,a.scope,a.limit,a.allow_draft)
    print(json.dumps({"status":"found" if results else "unknown","cards":results,"draft_mode":a.allow_draft,"audience":"model" if a.model_facing else "human_editorial"},ensure_ascii=False,indent=2))
if __name__=="__main__":main()
