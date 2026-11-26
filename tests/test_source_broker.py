import hashlib
import tempfile
import unittest
from pathlib import Path
from dataclasses import replace
from npr_gm.npr_bridge import git, BridgeError, blob_sha
from npr_gm.source_broker import SourceBroker, Grant, safe_path

class SourceBrokerTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        git(self.root,"init","-q","--template=")
        self.raw=b"secret header\\n".replace(b"\\n",b"\n")+b"public line\nprivate tail\n"
        (self.root/"note.txt").write_bytes(self.raw)
        git(self.root,"add","note.txt")
        git(self.root,"-c","user.name=Test","-c","user.email=test@example.invalid","commit","-qm","fixture")
        self.rev=git(self.root,"rev-parse","HEAD").decode().strip()
        self.grant=Grant("g1","note.txt",blob_sha(self.raw),2,2,hashlib.sha256(b"public line\n").hexdigest(),True)
    def broker(self,*grants,**kwargs):
        return SourceBroker(self.root,self.rev,grants or [self.grant],**kwargs)
    def test_exact_span_no_neighbor_or_path(self):
        self.assertEqual(self.broker().read("g1"),{"status":"found","id":"g1","text":"public line\n"})
    def test_dirty_checkout_ignored(self):
        (self.root/"note.txt").write_text("different")
        self.assertEqual(self.broker().read("g1")["text"],"public line\n")
    def test_hidden_and_absent_uniform(self):
        b=self.broker(replace(self.grant,player_visible=False))
        self.assertEqual(b.read("g1"),b.read("unknown"))
        self.assertEqual(b.read_ids,())
    def test_wrong_blob_and_excerpt(self):
        for g in (replace(self.grant,blob="f"*40),replace(self.grant,excerpt_sha256="f"*64),replace(self.grant,last=3)):
            self.assertEqual(self.broker(g).read("g1"),{"status":"unavailable"})
    def test_invalid_path_and_duplicate(self):
        for p in (".","..","../note.txt","/note.txt","a/../note.txt",".git/config","a\\b","a//b","a:1","-file"):
            self.assertFalse(safe_path(p),p)
        with self.assertRaises(BridgeError): self.broker(self.grant,self.grant)
    def test_shared_budget_and_citations(self):
        b=self.broker(budget=2)
        with self.assertRaises(BridgeError): b.require_citations(["g1"],["g1"])
        self.assertEqual(len(b.prefetch(["g1"])),1)
        self.assertEqual(b.remaining,1)
        b.require_citations(["g1"],["g1"])
        with self.assertRaises(BridgeError): b.require_citations(["g1","unknown"],["g1"])
        b.read("unknown")
        with self.assertRaises(BridgeError): b.read("g1")
    def test_prefetch_fails_closed_before_read(self):
        b=self.broker(budget=1)
        for ids in (["g1","unknown"],["unknown"],["g1","g1"]):
            with self.assertRaises(BridgeError): b.prefetch(ids)
            self.assertEqual(b.remaining,1)
            self.assertEqual(b.read_ids,())
    def test_unpinned_revision(self):
        with self.assertRaises(BridgeError): SourceBroker(self.root,"HEAD",[self.grant])

    def test_partial_prefetch_failure_poison(self):
        second=replace(self.grant,id="g2",excerpt_sha256="f"*64)
        b=self.broker(self.grant,second)
        with self.assertRaises(BridgeError): b.prefetch(["g1","g2"])
        self.assertEqual(b.read_ids,())
        with self.assertRaisesRegex(BridgeError,"source_broker_blocked"): b.read("g1")
        with self.assertRaisesRegex(BridgeError,"source_broker_blocked"): b.require_citations([],[])
