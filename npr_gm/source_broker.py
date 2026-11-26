"""Local exact-excerpt broker. Grants are trusted host decisions, never model input."""
from dataclasses import dataclass
import hashlib
import re
from pathlib import PurePosixPath, Path
from .npr_bridge import BridgeError, git, pin, blob_sha, SHA

ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")
DIGEST = re.compile(r"[0-9a-f]{64}")
MAX_BLOB = 1_000_000
MAX_EXCERPT = 8_000

@dataclass(frozen=True)
class Grant:
    id: str
    path: str
    blob: str
    first: int
    last: int
    excerpt_sha256: str
    player_visible: bool

def safe_path(path):
    if not isinstance(path,str) or not path or len(path)>512:
        return False
    if any(ord(c)<32 or ord(c)==127 or c in "\\:" for c in path):
        return False
    p=PurePosixPath(path)
    return not p.is_absolute() and p.as_posix()==path and all(x not in (".","..",".git") and not x.startswith("-") for x in p.parts)

class SourceBroker:
    """One broker per turn/snapshot. No path, line selector or revision from model.

    Grant visibility and exact excerpt digest must be reviewed by trusted host.
    No heuristic redaction of mixed private/public documents.
    """
    def __init__(self,root,revision,grants,*,budget=4):
        self._root=Path(root).resolve()
        self._revision=pin(self._root,revision)
        if type(budget) is not int or not 1<=budget<=8:
            raise BridgeError("invalid_source_budget")
        self._remaining=budget
        self._grants={}
        self._read={}
        for g in grants:
            if not isinstance(g,Grant) or not isinstance(g.id,str) or not ID.fullmatch(g.id) or g.id in self._grants:
                raise BridgeError("invalid_source_grant")
            if not safe_path(g.path) or not isinstance(g.blob,str) or not SHA.fullmatch(g.blob):
                raise BridgeError("invalid_source_grant")
            if not isinstance(g.excerpt_sha256,str) or not DIGEST.fullmatch(g.excerpt_sha256):
                raise BridgeError("invalid_source_grant")
            if type(g.first) is not int or type(g.last) is not int or not 1<=g.first<=g.last<=100_000 or g.last-g.first>=100:
                raise BridgeError("invalid_source_span")
            if type(g.player_visible) is not bool:
                raise BridgeError("invalid_source_visibility")
            self._grants[g.id]=g

    def _excerpt(self,g):
        entry=git(self._root,"ls-tree","-z",self._revision,"--",":(literal)"+g.path)
        rows=[x for x in entry.split(b"\0") if x]
        if len(rows)!=1: raise BridgeError("source_path_mismatch")
        meta,path=rows[0].split(b"\t",1)
        mode,kind,sha=meta.decode("ascii").split()
        if path.decode("utf-8")!=g.path or mode not in ("100644","100755") or kind!="blob" or sha!=g.blob:
            raise BridgeError("source_blob_mismatch")
        size=int(git(self._root,"cat-file","-s",sha))
        if not 0<size<=MAX_BLOB: raise BridgeError("source_blob_budget")
        raw=git(self._root,"cat-file","blob",sha)
        if len(raw)!=size or blob_sha(raw)!=sha: raise BridgeError("source_blob_mismatch")
        lines=raw.splitlines(keepends=True)
        if g.last>len(lines): raise BridgeError("source_span_mismatch")
        excerpt=b"".join(lines[g.first-1:g.last])
        if not 0<len(excerpt)<=MAX_EXCERPT or hashlib.sha256(excerpt).hexdigest()!=g.excerpt_sha256:
            raise BridgeError("source_excerpt_mismatch")
        return excerpt.decode("utf-8",errors="strict")

    @property
    def remaining(self):
        return self._remaining

    @property
    def read_ids(self):
        return tuple(sorted(self._read))

    def read(self,ident):
        if not self._remaining: raise BridgeError("source_budget_exhausted")
        self._remaining-=1
        g=self._grants.get(ident) if isinstance(ident,str) else None
        if g is None or not g.player_visible: return {"status":"unavailable"}
        try:
            text=self._excerpt(g)
        except (BridgeError,ValueError,OSError,TimeoutError):
            return {"status":"unavailable"}
        self._read[g.id]=g.excerpt_sha256
        return {"status":"found","id":g.id,"text":text}

    def prefetch(self,required):
        ids=tuple(required)
        if any(not isinstance(i,str) for i in ids) or len(set(ids))!=len(ids) or len(ids)>self._remaining:
            raise BridgeError("mandatory_source_budget_or_ids")
        # Fail before touching any source if one ID is not authorized.
        if any(i not in self._grants or not self._grants[i].player_visible for i in ids):
            raise BridgeError("mandatory_source_unavailable")
        results=[]
        for i in sorted(ids):
            r=self.read(i)
            if r["status"]!="found":
                # Caller must discard this broker/turn, not deliver a partial prompt.
                raise BridgeError("mandatory_source_unavailable")
            results.append(r)
        return results

    def require_citations(self,cited,required):
        if not isinstance(cited,list) or any(not isinstance(i,str) for i in cited) or len(set(cited))!=len(cited):
            raise BridgeError("invalid_source_citations")
        if not set(required)<=set(cited) or not set(cited)<=set(self._read):
            raise BridgeError("missing_or_unread_source")
