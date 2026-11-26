"""Paired public diagnostic; never training data or a live turn."""
import argparse
import collections
import statistics
import time
from .sandbox import Sandbox, Source, Outcome, ContractError, encode, parse_request
from .transport import SYSTEM, local_exchange

# Each source contains distinct information absent from outcome labels/context.
# Fixed before inference; no use of historical smoke, training drafts or private holdout.
CASES = (
    ("P01", "Изучаю табличку у причала.", ("Шлюз открывают в полдень.",), ("a17",)),
    ("P02", "Сверяю обе записи в журнале мастерской.", ("Мастер вернётся в четверг.", "Запасная печать лежит в синем футляре."), ("b21","b22")),
    ("P03", "Читаю карточку у музейного макета.", ("Макет изображает северную башню.",), ("c31",)),
    ("P04", "Проверяю расписание и отметку на квитанции.", ("Проверка назначена на рассвете.", "Квитанция действительна до пятницы."), ("d41","d42")),
    ("P05", "Осматриваю пустой подоконник.", (), ()),
    ("P06", "Изучаю ярлык на ящике.", ("Ящик предназначен для стекольной мастерской.",), ("f61",)),
)
BUDGET = 4
ROUNDS = 5

def fixture(case):
    key, player, texts, ids = case
    sources = [Source(i,t) for i,t in zip(ids,texts)]
    sources += [Source("hidden","SECRET_SENTINEL",False),Source("unrelated","UNRELATED_SENTINEL")]
    outcome=Outcome("o1",0,2,ids," ".join(texts) or "Подоконник пуст.")
    engine=Sandbox(resources=4,clock=100,sources=sources,outcomes=[outcome])
    return engine,engine.begin(player,["o1"],budget=BUDGET)

def run_arm(engine,turn,exchange,*,prefetch):
    """Same read budget and validation in both arms. No commit, no automatic citations."""
    context=engine.context(turn)
    for outcome in context["authorized_outcomes"]:
        del outcome["description"]
    required=sorted({i for o in context["authorized_outcomes"] for i in o["required_sources"]})
    prefetched=[]
    if prefetch:
        for i in required:
            result=engine.request(turn,encode({"tool":"read_source","id":i}))
            if result.get("status")!="found":
                raise ContractError("prefetch_unavailable")
            prefetched.append(result)
    context["prefetched_sources"]=prefetched
    messages=[{"role":"system","content":SYSTEM+" Все обязательные ID выбранного исхода должны быть в facts_used. Предзагруженные found-источники считаются прочитанными. Описания исходов скрыты; не выдумывай сведения."},
              {"role":"user","content":encode(context)}]
    calls=rounds=unavailable=0
    start=time.monotonic()
    stage="exchange"
    status="dialogue_round_limit"
    try:
        for _ in range(ROUNDS):
            rounds+=1
            stage="exchange"
            raw=exchange(list(messages))
            stage="parse"
            req=parse_request(raw)
            if "tool" in req:
                stage="read"
                calls+=1
                result=engine.request(turn,raw)
                unavailable+=result.get("status")=="unavailable"
                messages.extend([{"role":"assistant","content":raw},
                                 {"role":"user","content":encode({"tool_result":result,"instruction":"Продолжи по контракту."})}])
            else:
                stage="preview"
                engine.preview(turn,raw)
                status="valid"
                stage="complete"
                break
    except ContractError as e:
        status=str(e)
        if status=="missing_or_unread_source" and stage=="preview":
            cited=req.get("facts_used",[])
            status="missing_required_citation" if not set(required)<=set(cited) else "unread_or_extra_citation"
    except (OSError,ValueError,KeyError,IndexError,TypeError):
        status="exchange_or_response_error"
    # No raw messages, candidate or prose are published.
    return {"status":status,"stage":stage,"rounds":rounds,"model_reads":calls,
            "host_reads":len(prefetched),"unavailable_reads":unavailable,
            "elapsed_s":round(time.monotonic()-start,3),
            "unchanged":(engine.revision,engine.resources,engine.clock)==(turn.revision,context["resources"],context["clock"]) and engine.receipt(turn.id) is None}

def experiment(exchange):
    rows=[]
    for n,case in enumerate(CASES):
        # Counterbalance order to reduce systematic warm-up effects.
        for mode in (("direct","prefetch") if n%2==0 else ("prefetch","direct")):
            engine,turn=fixture(case)
            row=run_arm(engine,turn,exchange,prefetch=mode=="prefetch")
            rows.append({"case":case[0],"arm":mode,**row})
    summary={}
    for mode in ("direct","prefetch"):
        group=[r for r in rows if r["arm"]==mode]
        summary[mode]={"n":len(group),"valid":sum(r["status"]=="valid" for r in group),
                       "status_counts":dict(collections.Counter(r["status"] for r in group)),
                       "median_elapsed_s":statistics.median(r["elapsed_s"] for r in group),
                       "model_reads":sum(r["model_reads"] for r in group),
                       "host_reads":sum(r["host_reads"] for r in group)}
    pairs=collections.Counter()
    for case in CASES:
        pair={r["arm"]:r["status"]=="valid" for r in rows if r["case"]==case[0]}
        pairs["both" if all(pair.values()) else "direct_only" if pair["direct"] else "prefetch_only" if pair["prefetch"] else "neither"]+=1
    return {"schema":"prefetch-diagnostic/1","rows":rows,"summary":summary,"pairs":dict(pairs),
            "all_unchanged":all(r["unchanged"] for r in rows),"semantic_review":False,"training":False}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model",required=True)
    parser.add_argument("--endpoint",default="http://127.0.0.1:8080/v1/chat/completions")
    args=parser.parse_args()
    receipt=experiment(local_exchange(args.endpoint,args.model))
    print(encode(receipt))
    if not receipt["all_unchanged"] or any(r["status"]=="exchange_or_response_error" for r in receipt["rows"]):
        raise SystemExit(1)

if __name__=="__main__":
    main()
