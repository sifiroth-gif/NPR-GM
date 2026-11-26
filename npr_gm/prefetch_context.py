"""Experimental host context shared by diagnostic and training replay."""
from .sandbox import ContractError, encode
from .transport import SYSTEM

def initial_messages(engine,turn,*,prefetch):
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
    return context,required,prefetched,messages
