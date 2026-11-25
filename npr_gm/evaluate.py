"""Local-only evaluation, no writes to the game or external provider."""
import argparse
import json
import time
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit
from hashlib import sha256

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/synthetic_eval.jsonl'
FIELDS = {'scene', 'facts_used', 'actions', 'clock_advance'}
SYSTEM = ('Ты ведущий интерактивной RPG. Игрок управляет только собой. '
          'Соблюдай факты и ограничения текущей заявки, не раскрывай скрытые данные, '
          'не выдумывай результат броска. Верни только JSON с полями: '
          'scene (литературный текст), facts_used (список ID использованных фактов), '
          'actions (список ID подтвержденных действий), clock_advance (целое число минут). '
          'Используй только разрешенные ID из контекста. Не включай служебные поля в прозу.')

def load_cases(path=DATA):
    cases = [json.loads(s) for s in Path(path).read_text(encoding='utf-8').splitlines() if s.strip()]
    ids = [c['id'] for c in cases]
    if len(set(ids)) != len(ids) or not cases:
        raise ValueError('Duplicate or empty dataset')
    for c in cases:
        if c['split'] != 'eval' or c['source'] != 'synthetic' or not c['group']:
            raise ValueError('Unexpected provenance/split')
        if check(c, c['chosen']) or not check(c, c['rejected']):
            raise ValueError('Broken fixture: ' + c['id'])
    return cases

def check(case, answer):
    if not isinstance(answer, dict) or set(answer) != FIELDS:
        return ['schema']
    if not isinstance(answer['scene'], str) or not answer['scene'].strip():
        return ['scene']
    if not all(isinstance(answer[k], list) and all(isinstance(v, str) for v in answer[k]) for k in ('facts_used', 'actions')):
        return ['id_types']
    if type(answer['clock_advance']) is not int:
        return ['clock_type']
    errors=[]
    for k in ('facts_used','actions'):
        if len(set(answer[k])) != len(answer[k]): errors.append(k+'_duplicate')
    for field, allowed, required in [('facts_used','allowed_facts','required_facts'), ('actions','allowed_actions','required_actions')]:
        if not set(answer[field]) <= set(case[allowed]): errors.append(field+'_forbidden')
        if not set(case[required]) <= set(answer[field]): errors.append(field+'_missing')
    if not case['clock_min'] <= answer['clock_advance'] <= case['clock_max']: errors.append('clock_range')
    if any(s.casefold() in answer['scene'].casefold() for s in case['forbidden_text']): errors.append('forbidden_text')
    return errors

def messages(case):
    # No chosen/rejected text or rubric in model input.
    return [{'role':'system','content':SYSTEM}, {'role':'user','content':json.dumps({k:case[k] for k in ('context','player','allowed_facts','allowed_actions')}, ensure_ascii=False)}]

class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError('Redirects forbidden')

def generate(case, endpoint, model, timeout, *, max_tokens=1024, thinking=None):
    u=urlsplit(endpoint)
    if u.scheme != 'http' or u.hostname not in ('127.0.0.1','::1') or u.username or u.password or u.query or u.fragment or u.path.rstrip('/') != '/v1/chat/completions':
        raise ValueError('Only local llama.cpp chat endpoint allowed')
    payload={'model':model,'messages':messages(case),'temperature':0.2,'seed':42,'max_tokens':max_tokens,'stream':False}
    if thinking is not None:
        payload['chat_template_kwargs'] = {'enable_thinking': thinking}
    req=urllib.request.Request(endpoint, data=json.dumps(payload,ensure_ascii=False).encode(), headers={'Content-Type':'application/json'}, method='POST')
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}),NoRedirect())
    start=time.monotonic()
    with opener.open(req, timeout=timeout) as res:
        obj=json.loads(res.read(2_000_000))
    raw=obj['choices'][0]['message']['content']
    try: answer=json.loads(raw)
    except (ValueError,TypeError): answer=None
    return answer, raw, round(time.monotonic()-start,3), obj.get('usage',{})

def main():
    p=argparse.ArgumentParser(); p.add_argument('--check-data',action='store_true'); p.add_argument('--fixture',action='store_true'); p.add_argument('--model'); p.add_argument('--endpoint',default='http://127.0.0.1:8080/v1/chat/completions'); p.add_argument('--timeout',type=int,default=180); p.add_argument('--output',type=Path); a=p.parse_args(); cases=load_cases()
    if a.check_data:
        print(f'{len(cases)} cases; {len(set(c["group"] for c in cases))} groups; dataset sha256={sha256(DATA.read_bytes()).hexdigest()}'); return
    if not a.output or (not a.fixture and not a.model): p.error('Use --check-data or --output with --fixture/--model')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    passed=0
    with a.output.open('x',encoding='utf-8') as f:
        for case in cases:
            try:
                if a.fixture: answer,raw,elapsed,usage=case['chosen'],None,0,{}
                else: answer,raw,elapsed,usage=generate(case,a.endpoint,a.model,a.timeout)
                errors=check(case,answer)
            except Exception as e:
                answer,raw,elapsed,usage=None,None,None,{}; errors=['transport:'+type(e).__name__]
            passed+=not errors
            f.write(json.dumps({'case':case['id'],'group':case['group'],'kind':'fixture_NOT_INFERENCE' if a.fixture else 'inference','model':a.model,'dataset_sha256':sha256(DATA.read_bytes()).hexdigest(),'answer':answer,'raw':raw,'errors':errors,'seconds':elapsed,'usage':usage,'manual_review':None},ensure_ascii=False)+'\n'); f.flush()
    print(f'{passed}/{len(cases)} automatic checks passed; manual review not performed; '+('FIXTURE ONLY' if a.fixture else 'MODEL OUTPUT'))
    if a.fixture and passed != len(cases): raise SystemExit(1)
if __name__=='__main__': main()
