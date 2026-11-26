"""Validate public draft knowledge cards; not a medical accuracy validator."""
import json
from pathlib import Path
from urllib.parse import urlsplit
ROOT = Path(__file__).resolve().parents[1] / "data" / "knowledge"
DOMAINS = {"anatomy", "physiology", "neurophysiology", "psychophysiology", "medicine", "biology"}
def load():
    registry = json.loads((ROOT / "source_registry.json").read_text(encoding="utf-8"))
    cards = [json.loads(s) for s in (ROOT / "seed.jsonl").read_text(encoding="utf-8").splitlines() if s.strip()]
    return registry, cards
def validate(registry, cards):
    errors = []
    sources = registry["sources"]
    by_id = {x["id"]: x for x in sources}
    if len(by_id) != len(sources): errors.append("duplicate_source")
    if len({c["id"] for c in cards}) != len(cards): errors.append("duplicate_card")
    if {c["domain"] for c in cards} != DOMAINS: errors.append("domain_coverage")
    for c in cards:
        if c["source_id"] not in by_id: errors.append(c["id"] + ":unknown_source")
        else:
            u, root = urlsplit(c["url"]), urlsplit(by_id[c["source_id"]]["url"])
            if u.scheme != "https" or u.hostname != root.hostname: errors.append(c["id"] + ":source_host")
        if c["train_eligible"] is not False or c["use"] != "human_reference_review_only" or c["status"] != "draft_reference": errors.append(c["id"] + ":unreviewed_training")
        if c.get("model_use") not in {"blocked_pending_permission","pending_page_rights_and_expert_review"}: errors.append(c["id"] + ":draft_model_use")
        if c["source_id"] in by_id and by_id[c["source_id"]].get("model_use")=="blocked_pending_permission" and c.get("model_use")!="blocked_pending_permission": errors.append(c["id"] + ":source_permission_conflict")
        if not all(isinstance(c[k],str) and c[k].strip() for k in ("topic","fact","application","limitations","review","origin")): errors.append(c["id"] + ":missing_content")
    return errors
def main():
    registry, cards = load()
    errors = validate(registry, cards)
    print(f"{len(cards)} draft cards; {len(registry['sources'])} source records; {len({c['domain'] for c in cards})} domains; structural errors: {len(errors)}")
    if errors: raise SystemExit("; ".join(errors))
if __name__ == "__main__": main()
