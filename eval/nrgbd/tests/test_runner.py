import tempfile
import unittest
from pathlib import Path


class RunnerTests(unittest.TestCase):
    def test_atomic_commit_resume_and_partial(self):
        from nrgbd_eval.results import commit_scene, is_scene_complete, summarize

        with tempfile.TemporaryDirectory() as t:
            r = Path(t)
            prov = {"protocol": "x", "input": "a"}
            commit_scene(r, "one", {"acc": 1.0}, prov)
            self.assertTrue(is_scene_complete(r, "one", prov))
            self.assertFalse(
                is_scene_complete(r, "one", {"protocol": "y", "input": "a"})
            )
            from nrgbd_eval.results import require_resume_compatible

            with self.assertRaisesRegex(RuntimeError, "provenance"):
                require_resume_compatible(r, "one", {"protocol": "y", "input": "a"})
            fake = r / "scenes" / "two"
            fake.mkdir(parents=True)
            (fake / "metrics.json").write_text('{"acc": 9}')
            s = summarize(r, ["one", "two"])
            self.assertEqual(s["status"], "partial")
            self.assertEqual(s["missing"], ["two"])
            self.assertFalse(any(p.name.startswith(".staging") for p in r.iterdir()))
