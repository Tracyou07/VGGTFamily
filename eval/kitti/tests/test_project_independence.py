"""Standalone packaging and verification acceptance gates."""
import ast
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE = "kitti_eval"
SIBLINGS = {"kitti_eval", "waymo_eval", "virtual_kitti_eval", "scannet_eval"} - {MODULE}


def environment(root):
    return {**os.environ, "PYTHONPATH": str(root / "src"),
            "PYTHONDONTWRITEBYTECODE": "1", "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            "PYTEST_ADDOPTS": "", "CUDA_VISIBLE_DEVICES": ""}


def test_project_imports_and_cli_work_when_copied_alone(tmp_path):
    isolated = tmp_path / "standalone"
    shutil.copytree(ROOT, isolated, ignore=shutil.ignore_patterns(
        ".git", ".runtime", ".superpowers", "__pycache__", ".pytest_cache",
        "*.egg-info", "prepared", "results", "outputs", "runtime", "logs", "build", "dist"))
    code = """
import importlib, importlib.abc, pathlib, pkgutil, sys
class RejectSibling(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in SIBLINGS:
            raise AssertionError('sibling evaluator import: ' + fullname)
sys.meta_path.insert(0, RejectSibling())
package = importlib.import_module(MODULE)
assert pathlib.Path(package.__file__).resolve().is_relative_to(pathlib.Path('src').resolve())
for entry in pkgutil.walk_packages(package.__path__, package.__name__ + '.'):
    if not entry.name.endswith('.__main__'):
        importlib.import_module(entry.name)
assert not SIBLINGS.intersection(sys.modules)
"""
    code = f"MODULE = {MODULE!r}\nSIBLINGS = {SIBLINGS!r}\n" + code
    for arguments in (["-c", code], ["-m", MODULE, "--help"]):
        completed = subprocess.run([sys.executable, "-B", *arguments], cwd=isolated,
                                   env=environment(isolated), capture_output=True, text=True)
        assert completed.returncode == 0, completed.stdout + completed.stderr
    assert sorted(p.name for p in tmp_path.iterdir()) == ["standalone"]


def test_runtime_source_forbids_sibling_evaluator_imports():
    for path in (ROOT / "src").rglob("*.py"):
        tree = ast.parse(path.read_text())
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                imports.append(node.module)
            elif isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant):
                function = node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", "")
                if function in {"import_module", "__import__"} and isinstance(node.args[0].value, str):
                    imports.append(node.args[0].value)
        assert not {name.split(".")[0] for name in imports}.intersection(SIBLINGS), path


def test_project_has_no_stale_evaluator_paths():
    forbidden = ("/eval/" + "vggtlong", "/eval/" + "scannet/.runtime")
    paths = [ROOT / "README.md", ROOT / "pyproject.toml"]
    for directory in ("src", "configs", "scripts", "docs", "tests"):
        paths.extend(path for path in (ROOT / directory).rglob("*")
                     if path.is_file() and path.suffix in {".py", ".sh", ".json", ".txt", ".md", ".toml"})
    for path in paths:
        assert not any(value in path.read_text() for value in forbidden), path


def test_documentation_and_local_verification_entrypoint_are_present():
    readme = (ROOT / "README.md").read_text().lower()
    for topic in ("pip install", "data layout", "doctor", "prepare", "verify", "--model",
                  "--resume", "aggregate", "export-table", "timing boundary",
                  "peak allocated", "reserved", "mib",
                  "color_root", "aux_root", "model-owned long setup"):
        assert topic in readme, topic
    for filename in ("output-schema.md", "model-contract.md", "verification.md", "data-status.md"):
        assert (ROOT / "docs" / filename).is_file(), filename
        assert f"docs/{filename}" in readme, filename
    assert "scripts/verify_all_cpu.sh" in readme
    assert (ROOT / "scripts" / "verify_all_cpu.sh").is_file()


@pytest.mark.parametrize("behavior", ["success", "test_failure", "generated_artifact", "overwritten_ignored", "dirty_tracked"])
def test_verification_script_is_local_and_propagates_failures(tmp_path, behavior):
    script = ROOT / "scripts" / "verify_all_cpu.sh"
    assert script.is_file()
    project = tmp_path / "project"
    (project / "scripts").mkdir(parents=True)
    (project / "tests").mkdir()
    (project / "src" / MODULE).mkdir(parents=True)
    shutil.copyfile(script, project / "scripts" / script.name)
    helper = ROOT / "scripts" / "snapshot_repository.py"
    if helper.exists():
        shutil.copyfile(helper, project / "scripts" / helper.name)
    (project / "src" / MODULE / "__init__.py").write_text("")
    (project / "src" / MODULE / "__main__.py").write_text("print('standalone help')\n")
    (project / ".gitignore").write_text("results/\n")
    bodies = {
        "success": "def test_ok():\n    assert True\n",
        "test_failure": "def test_failure():\n    assert False\n",
        "generated_artifact": (
            "from pathlib import Path\n"
            "def test_leaked_output():\n"
            "    Path('results').mkdir()\n"
            "    Path('results/leaked.json').write_text('{}')\n"),
    }
    for kind, target in (("overwritten_ignored", "results/leaked.json"),
                         ("dirty_tracked", "tracked.txt")):
        bodies[kind] = ("import os\nfrom pathlib import Path\n"
            "def test_overwrite():\n"
            f"    path = Path({target!r})\n"
            "    before = path.stat()\n"
            "    path.write_bytes(b'after!')\n"
            "    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))\n")
    if behavior == "overwritten_ignored":
        (project / "results").mkdir()
        (project / "results/leaked.json").write_bytes(b"prior!")
    if behavior == "dirty_tracked":
        (project / "tracked.txt").write_bytes(b"clean!")
    (project / "tests" / "test_probe.py").write_text(bodies[behavior])
    # A sibling test must never be discovered by this repository's entry point.
    sibling = tmp_path / "sibling"
    sibling.mkdir()
    (sibling / "test_forbidden.py").write_text("raise AssertionError('sibling executed')\n")
    subprocess.run(["git", "init", "-q", str(project)], check=True)
    subprocess.run(["git", "add", "."], cwd=project, check=True)
    if behavior == "dirty_tracked":
        (project / "tracked.txt").write_bytes(b"prior!")
    before_status = subprocess.check_output(["git", "status", "--porcelain=v1",
        "--untracked-files=all", "--ignored"], cwd=project, text=True)
    completed = subprocess.run(["bash", str(project / "scripts" / script.name)],
        cwd=tmp_path, env={**environment(project), "PYTHON": sys.executable},
        capture_output=True, text=True)
    output = completed.stdout + completed.stderr
    if behavior in {"overwritten_ignored", "dirty_tracked"}:
        after_status = subprocess.check_output(["git", "status", "--porcelain=v1",
            "--untracked-files=all", "--ignored"], cwd=project, text=True)
        assert before_status == after_status  # The old status-only gate cannot see this mutation.
    assert "sibling executed" not in output
    if behavior == "success":
        assert completed.returncode == 0, output
        assert "standalone help" not in output
    else:
        assert completed.returncode != 0, output
        if behavior in {"generated_artifact", "overwritten_ignored", "dirty_tracked"}:
            assert "Generated repository changes" in output


def snapshot(root):
    helper = ROOT / "scripts" / "snapshot_repository.py"
    assert helper.is_file()
    completed = subprocess.run([sys.executable, "-B", str(helper), str(root)],
                               env=environment(root), capture_output=True, text=True, timeout=10)
    assert completed.returncode == 0, completed.stderr
    return completed.stdout


def test_snapshot_records_identity_even_when_replacement_bytes_match(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    target = root / "source.py"
    target.write_bytes(b"unchanged bytes")
    before = snapshot(root)
    replacement = root / "replacement"
    replacement.write_bytes(target.read_bytes())
    replacement.replace(target)
    assert snapshot(root) != before


def test_snapshot_ignores_external_symlink_contents_and_explicit_caches(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    external = tmp_path / "external-models-and-data"
    external.mkdir()
    (external / "large-model.bin").write_bytes(b"before")
    (root / "external").symlink_to(external, target_is_directory=True)
    (root / "external-file").symlink_to(external / "large-model.bin")
    (root / ".runtime").mkdir()
    (root / "source.py").write_text("value = 1\n")
    before = snapshot(root)
    (external / "large-model.bin").write_bytes(b"modified outside repository")
    for cache in (".git", ".superpowers", ".pytest_cache", "__pycache__",
                  ".mypy_cache", ".ruff_cache", ".runtime/long_deps"):
        path = root / cache / "changed"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("benign excluded scratch")
    (root / "generated.pyc").write_bytes(b"bytecode")
    assert snapshot(root) == before
