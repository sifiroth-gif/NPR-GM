import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from npr_gm.npr_bridge import inspect_anchor,BridgeError,blob_sha,check,read_snapshot

def packet(sha):
    return {"schema":"npr-next/1","cursor":"s01.text.001","next_id":"s01.text.002","turn_file":sha,
            "lag":{"recorded_turn":"s01.text.001"},"fold_errors":[]}
def turn(n,prior=None):
    return {"schema":"npr-turn/2","id":f"s01.text.{n:03}","review":{"predecessor":prior},
            "player":"Проверка","gm":"Ответ","kind":"ooc","at":"0010-01-01T10:00:00"}
class AnchorTests(unittest.TestCase):
    def test_fresh(self):
        self.assertEqual(inspect_anchor(packet("a"*40),turn(1),"a"*40,[])["mode"],"fresh")
    def test_same_id_changed_blob(self):
        with self.assertRaises(BridgeError):inspect_anchor(packet("a"*40),turn(1),"b"*40,[])
    def test_lag_without_receipt(self):
        with self.assertRaises(BridgeError):inspect_anchor(packet("a"*40),turn(2),"b"*40,[])
    def test_blocked(self):
        p=packet("a"*40);p["fold_errors"]=["x"]
        with self.assertRaises(BridgeError):inspect_anchor(p,turn(1),"a"*40,[])
    def test_receipt_chain(self):
        t=turn(2,"s01.text.001")
        self.assertEqual(inspect_anchor(packet("a"*40),t,"b"*40,[{"turn":t,"sha":"b"*40}])["mode"],"receipt_projection")
    def test_missing_latest(self):
        t=turn(2,"s01.text.001")
        with self.assertRaises(BridgeError):inspect_anchor(packet("a"*40),turn(3),"c"*40,[{"turn":t,"sha":"b"*40}])
    def test_amended_receipt(self):
        t=turn(2,"s01.text.001");t["amend"]=True
        with self.assertRaises(BridgeError):inspect_anchor(packet("a"*40),t,"b"*40,[{"turn":t,"sha":"b"*40}])
    def test_recorded_anchor(self):
        p=packet("a"*40);p["lag"]["recorded_turn"]="s01.text.002"
        t=turn(2,"s01.text.001")
        with self.assertRaises(BridgeError):inspect_anchor(p,t,"b"*40,[{"turn":t,"sha":"b"*40}])

class LocalGitTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.git("init","-q");self.git("config","user.name","Synthetic");self.git("config","user.email","synthetic@example.invalid")
        self.write("rpg_engine/__init__.py","")
        # Synthetic stub, explicitly not original NPR source or mechanics.
        self.write("rpg_engine/overlay.py","def project(p, r):\n return dict(p, cursor=r[-1]['turn']['id'], turn_file=r[-1]['sha']) if r else dict(p)\n")
        self.write("rpg_engine/preflight.py","def inspect(p,d,receipts=(),used=()):\n return {'mechanical_ok':d['review']['predecessor']==(receipts[-1]['turn']['id'] if receipts else p['cursor']),'errors':[],'review':[],'checklist':[1,2,3,4,5,6]}\n")
        self.code=self.commit()
        raw=json.dumps(turn(1)).encode()
        self.write("recovery/TURN.json",raw.decode());self.write("recovery/NEXT.json",json.dumps(packet(blob_sha(raw))))
        self.base=self.commit()
    def git(self,*args):
        return subprocess.check_output(["git","-C",str(self.root),*args],stderr=subprocess.DEVNULL).decode().strip()
    def write(self,path,content):
        p=self.root/path;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(content,encoding="utf-8")
    def commit(self):
        self.git("add",".");self.git("commit","-qm","fixture");return self.git("rev-parse","HEAD")
    def test_original_bridge_boundary_with_stub(self):
        before=self.git("rev-parse","HEAD")
        result=check(self.root,self.code,self.base,turn(2,"s01.text.001"))
        self.assertTrue(result["mechanical_ok"]);self.assertFalse(result["committed"])
        self.assertEqual(before,self.git("rev-parse","HEAD"));self.assertEqual(self.git("status","--porcelain"),"")
    def test_pinned_not_dirty_checkout(self):
        self.write("rpg_engine/preflight.py","raise RuntimeError('dirty')\n")
        self.assertTrue(check(self.root,self.code,self.base,turn(2,"s01.text.001"))["mechanical_ok"])
    def test_verified_receipt_from_git(self):
        self.write("recovery/TURN.json",json.dumps(turn(2,"s01.text.001")))
        ahead=self.commit()
        with self.assertRaises(BridgeError):read_snapshot(self.root,ahead)
        result=check(self.root,self.code,ahead,turn(3,"s01.text.002"),receipt_commits=[ahead])
        self.assertTrue(result["mechanical_ok"]);self.assertEqual(result["snapshot_mode"],"receipt_projection")
    def test_no_unpinned_ref(self):
        with self.assertRaises(BridgeError):check(self.root,"HEAD",self.base,turn(2))
