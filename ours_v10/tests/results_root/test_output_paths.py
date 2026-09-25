from __future__ import annotations

from pathlib import Path
import unittest

from scripts.autodl.migrate_results_root import EXPERIMENT_ROOTS


ROOT = Path(__file__).resolve().parents[2]


class CanonicalOutputPathTest(unittest.TestCase):
    def test_executable_sources_have_no_legacy_experiment_defaults(self):
        paths = sorted((ROOT / "pre_experiments").rglob("*.py"))
        paths.extend(sorted((ROOT / "scripts" / "autodl").rglob("*.py")))
        paths.extend(sorted((ROOT / "scripts" / "autodl").rglob("*.sh")))
        violations = []
        for path in paths:
            content = path.read_text(encoding="utf-8")
            for experiment in EXPERIMENT_ROOTS:
                patterns = (
                    f"$AUTODL_TMP/{experiment}",
                    f"${{AUTODL_TMP}}/{experiment}",
                    f'/root/autodl-tmp/{experiment}',
                    f'AUTODL_TMP / "{experiment}"',
                    f"AUTODL_TMP / '{experiment}'",
                )
                for pattern in patterns:
                    if pattern in content:
                        violations.append(
                            f"{path.relative_to(ROOT).as_posix()}: {pattern}"
                        )
        self.assertEqual(violations, [])


if __name__ == "__main__":
    unittest.main()
