"""Read-only lexical retrieval for draft knowledge cards; not clinical advice."""
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

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("query")
    p.add_argument("--domain",choices=sorted(DOMAINS))
    p.add_argument("--scope")
    p.add_argument("--limit",type=int,default=3)
    p.add_argument("--allow-draft",action="store_true")
    a=p.parse_args()
    registry,cards=load()
    if validate(registry,cards): raise SystemExit("Knowledge corpus integrity failure")
    results=retrieve(a.query,cards,a.domain,a.scope,a.limit,a.allow_draft)
    print(json.dumps({"status":"found" if results else "unknown","cards":results,"draft_mode":a.allow_draft},ensure_ascii=False,indent=2))
if __name__=="__main__":main()
