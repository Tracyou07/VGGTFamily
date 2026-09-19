import ast
import unittest
from pathlib import Path


class ContractTests(unittest.TestCase):
    def test_backends_do_not_import_scoring(self):
        root = Path(__file__).parents[1] / "src" / "nrgbd_eval" / "backends"
        for p in root.glob("*.py"):
            tree = ast.parse(p.read_text())
            names = [
                n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)
            ]
            self.assertFalse(any("scoring" in x for x in names), p)

    def test_native_bridge_invariants(self):
        root = Path(__file__).parents[1] / "src" / "nrgbd_eval" / "native"
        self.assertIn("im.unsqueeze(0)", (root / "streamvggt.py").read_text())
        slam = (root / "vggt_slam.py").read_text()
        self.assertIn("get_points_list_in_world_frame", slam)
        self.assertIn("shared_retrieval", slam)
        self.assertIn("VGGT_Long", (root / "vggt_long.py").read_text())
