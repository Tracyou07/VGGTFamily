import json
import tempfile
import unittest
import subprocess
import sys
from pathlib import Path
from helpers import make_dataset


class CliTests(unittest.TestCase):
    def test_check_json(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            make_dataset(root / "data")
            cfg = root / "c.json"
            cfg.write_text(
                json.dumps(
                    {
                        "dataset_root": str((root / "data").resolve()),
                        "output_root": str((root / "out").resolve()),
                        "protocol": {"kf": 10},
                    }
                )
            )
            p = subprocess.run(
                [sys.executable, "-m", "nrgbd_eval", "check", "--config", str(cfg)],
                text=True,
                capture_output=True,
            )
            self.assertEqual(p.returncode, 0, p.stderr)
            self.assertTrue(json.loads(p.stdout)["ready"])
