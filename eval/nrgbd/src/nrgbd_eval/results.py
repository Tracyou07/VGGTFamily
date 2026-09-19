from pathlib import Path
import json
import os
import shutil
import tempfile


def _write(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with path.open("r+b") as f:
        os.fsync(f.fileno())


def _fsync_directory(path):
    flags = getattr(os, "O_DIRECTORY", None)
    if flags is None:
        return
    directory_fd = os.open(path, os.O_RDONLY | flags)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def commit_scene(run_root, scene_id, metrics, provenance, metadata=None):
    root = Path(run_root)
    scenes = root / "scenes"
    scenes.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".staging-{scene_id}-", dir=scenes))
    _write(stage / "metrics.json", metrics)
    _write(stage / "provenance.json", provenance)
    _write(stage / "prediction_metadata.json", metadata or {})
    complete = stage / "COMPLETE"
    complete.write_text("complete\n")
    with complete.open("r+b") as handle:
        os.fsync(handle.fileno())
    _fsync_directory(stage)
    target = scenes / scene_id
    if target.exists():
        shutil.rmtree(target)
    os.replace(stage, target)


def is_scene_complete(root, scene_id, provenance):
    p = Path(root) / "scenes" / scene_id
    if not (p / "COMPLETE").is_file():
        return False
    try:
        return json.loads((p / "provenance.json").read_text()) == provenance
    except Exception:
        return False


def require_resume_compatible(root, scene_id, provenance):
    scene = Path(root) / "scenes" / scene_id
    if not (scene / "COMPLETE").is_file():
        return False
    if not is_scene_complete(root, scene_id, provenance):
        raise RuntimeError(
            f"completed scene {scene_id} has incompatible provenance; use a new output"
        )
    return True


def summarize(root, expected):
    root = Path(root)
    rows = []
    missing = []
    for s in expected:
        scene_dir = root / "scenes" / s
        p = scene_dir / "metrics.json"
        if p.is_file() and (scene_dir / "COMPLETE").is_file():
            rows.append(json.loads(p.read_text()))
        else:
            missing.append(s)
    keys = (
        "acc",
        "acc_med",
        "comp",
        "comp_med",
        "nc1",
        "nc1_med",
        "nc2",
        "nc2_med",
        "nc",
        "nc_med",
    )
    summary = {
        k: sum(float(r[k]) for r in rows) / len(rows)
        for k in keys
        if rows and all(k in r for r in rows)
    }
    summary.update(
        status="complete" if not missing else "partial",
        completed=len(rows),
        expected=len(expected),
        missing=missing,
    )
    _write(root / "summary.json", summary)
    return summary
