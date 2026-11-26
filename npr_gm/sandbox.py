"""Experimental, in-memory GM contract. Not the NPR production TURN schema.

Only synthetic trusted fixtures or deliberately isolated data may be used here.
Model requests never execute code, select arbitrary paths, or supply state deltas.
"""
from dataclasses import dataclass
import hashlib
import json

class ContractError(ValueError):
    pass

def encode(obj):
    return json.dumps(obj,ensure_ascii=False,sort_keys=True,separators=(",",":"))

def parse_request(raw):
    if not isinstance(raw,str) or len(raw.encode("utf-8"))>12000:
        raise ContractError("invalid_request_size")
    def unique(pairs):
        obj={}
        for k,v in pairs:
            if k in obj: raise ContractError("duplicate_key")
            obj[k]=v
        return obj
    try:
        req=json.loads(raw,object_pairs_hook=unique,parse_constant=lambda _: (_ for _ in ()).throw(ContractError("invalid_number")))
    except (ValueError,RecursionError) as e:
        raise ContractError("invalid_json") from e
    if not isinstance(req,dict): raise ContractError("invalid_request")
    return req

@dataclass(frozen=True)
class Source:
    id:str
    text:str
    player_visible:bool=True

@dataclass(frozen=True)
class Outcome:
    id:str
    cost:int
    minutes:int
    required_sources:tuple
    public_text:str

@dataclass(frozen=True)
class Turn:
    id:str
    revision:int
    player:str
    outcome_ids:tuple
    source_ids:tuple

class Sandbox:
    def __init__(self, *, resources, clock, sources, outcomes):
        if type(resources) is not int or resources<0 or type(clock) is not int or clock<0:
            raise ContractError("invalid_state")
        self.resources=resources
        self.clock=clock
        self.revision=0
        self._sources={s.id:s for s in sources}
        self._outcomes={o.id:o for o in outcomes}
        if len(self._sources)!=len(sources) or len(self._outcomes)!=len(outcomes):
            raise ContractError("duplicate_fixture_id")
        for o in outcomes:
            if type(o.cost) is not int or o.cost<0 or type(o.minutes) is not int or o.minutes<0:
                raise ContractError("invalid_trusted_outcome")
            if any(k not in self._sources or not self._sources[k].player_visible for k in o.required_sources):
                raise ContractError("invalid_public_source")
        self._active={}
        self._receipts={}
        self._serial=0

    def begin(self, player, authorized_outcomes, *, authorized_sources=None, budget=4):
        # Called by the trusted host only, after interpretation and mechanic resolution.
        if not isinstance(player,str) or not player.strip() or len(player)>4000:
            raise ContractError("invalid_player")
        if type(budget) is not int or not 1<=budget<=8:
            raise ContractError("invalid_budget")
        ids=tuple(authorized_outcomes)
        if not ids or len(ids)!=len(set(ids)) or any(k not in self._outcomes for k in ids):
            raise ContractError("invalid_authorizations")
        required={k for oid in ids for k in self._outcomes[oid].required_sources}
        permitted=set(required if authorized_sources is None else authorized_sources)
        if not required<=permitted or any(k not in self._sources or not self._sources[k].player_visible for k in permitted):
            raise ContractError("invalid_source_authorizations")
        self._serial+=1
        turn=Turn("t"+str(self._serial),self.revision,player,ids,tuple(sorted(permitted)))
        self._active[turn.id]={"turn":turn,"remaining":budget,"read":set()}
        return turn

    def _session(self, turn):
        session=self._active.get(turn.id)
        if not session or session["turn"]!=turn or turn.revision!=self.revision:
            raise ContractError("stale_or_closed_turn")
        return session

    def context(self,turn):
        self._session(turn)
        return {"schema":"npr-gm-sandbox/1","turn_id":turn.id,"revision":turn.revision,
                "player":turn.player,"resources":self.resources,"clock":self.clock,
                "source_ids":list(turn.source_ids),
                "authorized_outcomes":[{"id":k,"required_sources":list(self._outcomes[k].required_sources),
                  "cost":self._outcomes[k].cost,"minutes":self._outcomes[k].minutes,
                  "description":self._outcomes[k].public_text} for k in turn.outcome_ids]}

    def request(self,turn,raw):
        session=self._session(turn)
        if session["remaining"]<=0: raise ContractError("tool_budget_exhausted")
        # Invalid attempts count too. One final proposal is separate from the read budget.
        session["remaining"]-=1
        req=parse_request(raw)
        if set(req)!={"tool","id"} or req["tool"]!="read_source" or not isinstance(req["id"],str):
            raise ContractError("unsupported_tool")
        s=self._sources.get(req["id"])
        # Uniform miss for secret and absent IDs; never return hidden metadata.
        if s is None or not s.player_visible or s.id not in turn.source_ids: return {"status":"unavailable"}
        session["read"].add(s.id)
        return {"status":"found","id":s.id,"text":s.text}

    def preview(self,turn,raw):
        session=self._session(turn)
        req=parse_request(raw)
        if set(req)!={"schema","turn_id","revision","outcome_id","facts_used","scene_draft"}:
            raise ContractError("invalid_candidate_fields")
        if req["schema"]!="npr-gm-sandbox/1" or req["turn_id"]!=turn.id or type(req["revision"]) is not int or req["revision"]!=self.revision:
            raise ContractError("stale_or_invalid_candidate")
        oid=req["outcome_id"]
        if not isinstance(oid,str) or oid not in turn.outcome_ids:
            raise ContractError("unauthorized_outcome")
        facts=req["facts_used"]
        if not isinstance(facts,list) or not all(isinstance(k,str) for k in facts) or len(facts)!=len(set(facts)):
            raise ContractError("invalid_facts")
        o=self._outcomes[oid]
        if not set(facts)<=session["read"] or not set(o.required_sources)<=set(facts):
            raise ContractError("missing_or_unread_source")
        if o.cost>self.resources: raise ContractError("insufficient_resource")
        if not isinstance(req["scene_draft"],str) or not req["scene_draft"].strip() or len(req["scene_draft"])>4000:
            raise ContractError("invalid_draft")
        return {"outcome":o,"draft":req["scene_draft"],"facts":tuple(facts),"hash":hashlib.sha256(encode(req).encode()).hexdigest()}

    def commit(self,turn,raw):
        # This is an in-memory lab transaction, not durable CAS or a live turn.
        p=self.preview(turn,raw)
        o=p["outcome"]
        self.resources-=o.cost
        self.clock+=o.minutes
        self.revision+=1
        receipt={"turn_id":turn.id,"revision":self.revision,"candidate_sha256":p["hash"],
                 "outcome_id":o.id,"resources":self.resources,"clock":self.clock,
                 "player_text":o.public_text,"draft_status":"unreviewed_not_published"}
        self._receipts[turn.id]=receipt.copy()
        self._active.clear() # all concurrent proposals are now stale
        return receipt

    def receipt(self,turn_id):
        r=self._receipts.get(turn_id)
        return r.copy() if r is not None else None

def demo():
    engine=Sandbox(resources=4,clock=100,sources=[
        Source("notice","Выставка закрыта до утра."),
        Source("internal","Внутренняя запись.",False)],
        outcomes=[Outcome("inspect_notice",0,2,("notice",),"На двери висит объявление: выставка закрыта до утра."),
                  Outcome("reserve_ticket",2,3,(),"Билет забронирован.")])
    turn=engine.begin("Читаю объявление.",["inspect_notice"])
    read=engine.request(turn,encode({"tool":"read_source","id":"notice"}))
    assert read["status"]=="found"
    candidate={"schema":"npr-gm-sandbox/1","turn_id":turn.id,"revision":turn.revision,
               "outcome_id":"inspect_notice","facts_used":["notice"],"scene_draft":"На двери объявление."}
    return engine.commit(turn,encode(candidate))

if __name__=="__main__":
    print(encode(demo()))
