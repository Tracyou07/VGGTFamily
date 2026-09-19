import tempfile
import unittest
from pathlib import Path


class BackendTests(unittest.TestCase):
    def test_five_backends_allocation_free_and_doctor(self):
        from nrgbd_eval.backends import NAMES, create_backend, doctor_backend

        self.assertEqual(
            set(NAMES), {"vggt", "vggt_long", "streamvggt", "vggt_slam", "vggt_omega"}
        )
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            src = root / "src"
            src.mkdir()
            ck = root / "m.pt"
            ck.write_bytes(b"x")
            cfg = {
                n: {
                    "source_root": str(src.resolve()),
                    "checkpoint": str(ck.resolve()),
                    "python": "/usr/bin/python3",
                }
                for n in NAMES
            }
            for n in NAMES:
                b = create_backend(n, cfg[n], "cuda:0")
                self.assertEqual(b.name, n)
                self.assertTrue(doctor_backend(n, cfg[n])["ready"])
                first = b.provenance()
                ck.write_bytes(ck.read_bytes() + n.encode())
                b2 = create_backend(n, cfg[n], "cuda:0")
                self.assertNotEqual(
                    first["checkpoint_sha256"], b2.provenance()["checkpoint_sha256"]
                )

    def test_request_contains_no_gt(self):
        from nrgbd_eval.contracts import ModelSceneInput

        m = ModelSceneInput("x", ("1",), (Path("/a").resolve(),))
        self.assertNotIn("depth", repr(m).lower())
        self.assertNotIn("pose", repr(m).lower())
