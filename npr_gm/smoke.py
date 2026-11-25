"""A real, bounded smoke test. Never output raw scenes to public CI."""
import argparse
import json
import statistics
from hashlib import sha256
from .evaluate import load_cases, check, generate, DATA

def summarize(rows):
    responses=[r for r in rows if r["seconds"] is not None]
    return {"attempted":len(rows), "responses":len(responses),
            "automatic_pass":sum(not r["errors"] for r in rows),
            "median_response_seconds":statistics.median([r["seconds"] for r in responses]) if responses else None,
            "results":rows, "manual_review":"not_performed"}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model",required=True)
    p.add_argument("--endpoint",default="http://127.0.0.1:8080/v1/chat/completions")
    p.add_argument("--cases",nargs="+",default=["E01","E09","E19"])
    p.add_argument("--timeout",type=int,default=180)
    p.add_argument("--max-tokens",type=int,default=384)
    a=p.parse_args()
    cases={c["id"]:c for c in load_cases()}
    if len(set(a.cases))!=len(a.cases) or not all(k in cases for k in a.cases):
        p.error("Unknown or duplicate case ID")
    if not 1<=a.max_tokens<=2048 or not 1<=a.timeout<=300:
        p.error("Out of bounded smoke range")
    rows=[]
    for k in a.cases:
        try:
            ans,raw,seconds,usage=generate(cases[k],a.endpoint,a.model,a.timeout,max_tokens=a.max_tokens,thinking=False)
            errors=check(cases[k],ans)
        except Exception as e:
            seconds,usage,errors=None,{},["transport:"+type(e).__name__]
        row={"id":k,"seconds":seconds,"errors":errors,
             "usage":{x:usage[x] for x in ("prompt_tokens","completion_tokens","total_tokens") if type(usage.get(x)) is int}}
        rows.append(row)
        print(json.dumps(row),flush=True)
    result=summarize(rows)
    result.update({"model":a.model,"dataset_sha256":sha256(DATA.read_bytes()).hexdigest(),
                   "kind":"REAL_SMOKE","max_tokens":a.max_tokens,"thinking":False})
    print(json.dumps(result,ensure_ascii=False),flush=True)
    if result["responses"]!=result["attempted"]:
        raise SystemExit(2)
if __name__=="__main__":main()
