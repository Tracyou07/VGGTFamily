"""Read-only ScanNet selection contracts and guarded v10 run bookkeeping."""
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import stat

import numpy as np


EVAL_ROOT = Path("/home/ubuntu/yjh/feedforwardreconstruct/eval/scannet")
if str(EVAL_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(EVAL_ROOT))

from scannet_eval.data import read_scene_list  # noqa: E402
from scannet_eval.runner import METRIC_KEYS, SCENE_METRIC_KEYS  # noqa: E402
from scannet_eval.sens import sample_frame_ids  # noqa: E402


BUDGETS = (100, 300, 500, 1000)
SCENE_LIST = EVAL_ROOT / "configs" / "scannet50.txt"
PREPARED_ROOT = Path("/data/yjh/share/datasets/ScanNet/prepared_scannet50_v1")
BASELINE_ROOT = EVAL_ROOT / "results"
OUTPUT_BASE = Path("/data/yjh/output/vggt")
SCRATCH_BASE = Path("/home/ubuntu/yjh/feedforwardreconstruct/ours_v10_experiments")
SCENE_PATTERN = re.compile(r"^scene\d{4}_\d{2}$")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_metrics(values):
    if not isinstance(values, dict) or set(values) != set(SCENE_METRIC_KEYS):
        raise ValueError("missing or extra ScanNet scene metrics")
    result = {}
    for key in SCENE_METRIC_KEYS:
        value = values[key]
        if isinstance(value, bool):
            raise ValueError(f"invalid metric {key}")
        try:
            result[key] = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"invalid metric {key}") from error
        if not np.isfinite(result[key]):
            raise ValueError(f"nonfinite metric {key}")
    return result


def audit_frame_selection(scene_ids, budgets, prepared_root=PREPARED_ROOT,
                          baseline_root=BASELINE_ROOT):
    """Check original valid-frame selection against only complete VGGT* baselines."""
    rows = []
    for budget in budgets:
        if budget not in BUDGETS:
            raise ValueError("unsupported ScanNet frame budget")
        baseline = Path(baseline_root) / f"vggt_star_f{budget}_scannet50"
        summary = read_json(baseline / "summary.json")
        if summary.get("complete") is not True:
            raise ValueError(f"VGGT* f{budget} baseline summary is not complete")
        for scene_id in scene_ids:
            if not SCENE_PATTERN.fullmatch(scene_id):
                raise ValueError(f"invalid scene ID: {scene_id}")
            manifest = read_json(Path(prepared_root) / scene_id / "manifest.json")
            original = manifest["frame_ids"]
            if (not original or any(not isinstance(x, int) or isinstance(x, bool) for x in original)
                    or original != sorted(set(original))):
                raise ValueError(f"invalid prepared frame IDs: {scene_id}")
            selected = list(sample_frame_ids(original, budget))
            reference = read_json(baseline / scene_id / "result.json")
            if selected != reference.get("frame_ids"):
                raise ValueError(f"VGGT* selected frame IDs differ: {scene_id} f{budget}")
            rows.append(dict(scene_id=scene_id, frame_budget=budget,
                             actual_frames=len(selected), frame_ids=selected,
                             baseline_summary_complete=True))
    return rows


def _within_and_no_symlink(base, path):
    base, path = Path(base).absolute(), Path(path).absolute()
    try:
        relative = path.relative_to(base)
    except ValueError as error:
        raise ValueError(f"path outside allowed root: {path}") from error
    for part in (base, *[base.joinpath(*relative.parts[:i])
                           for i in range(1, len(relative.parts) + 1)]):
        if part.is_symlink():
            raise ValueError(f"symlink in owned path: {part}")
    if not path.resolve().is_relative_to(base.resolve()):
        raise ValueError(f"resolved path outside allowed root: {path}")
    return path


@dataclass(frozen=True)
class RunPaths:
    output: Path
    scratch: Path


def safe_run_paths(output_root, scratch_root, scene_id, budget, *,
                   allowed_output_base=OUTPUT_BASE, allowed_scratch_base=SCRATCH_BASE):
    if not isinstance(scene_id, str) or not SCENE_PATTERN.fullmatch(scene_id):
        raise ValueError("invalid scene ID")
    if budget not in BUDGETS:
        raise ValueError("unsupported ScanNet frame budget")
    output = _within_and_no_symlink(allowed_output_base,
                                   Path(output_root) / f"f{budget}" / scene_id)
    scratch = _within_and_no_symlink(allowed_scratch_base,
                                    Path(scratch_root) / f"f{budget}" / scene_id)
    if output == scratch or output.is_relative_to(scratch) or scratch.is_relative_to(output):
        raise ValueError("result and scratch paths overlap")
    return RunPaths(output, scratch)


def _validated_final(output):
    output = Path(output)
    summary = read_json(output / "summary.json")
    if summary.get("complete") is not True:
        raise ValueError("scene summary is not complete; cleanup forbidden")
    metrics = validate_metrics(read_json(output / "metrics.json"))
    result = read_json(output / "result.json")
    if validate_metrics(result.get("metrics")) != metrics:
        raise ValueError("result/metrics disagree")
    for name in ("run_manifest.json", "forward_manifest.json", "stitch_manifest.json",
                 "trajectory.npz", "alignment_edges.json"):
        if not (output / name).is_file():
            raise ValueError(f"missing retained result artifact: {name}")
    for name, key in (("metrics.json", "metrics_sha256"),
                      ("result.json", "result_sha256"),
                      ("trajectory.npz", "trajectory_sha256"),
                      ("alignment_edges.json", "alignment_edges_sha256")):
        if key in summary and sha256(output / name) != summary[key]:
            raise ValueError(f"confirmed result artifact changed: {name}")
    return metrics


def _regenerable(path, scratch):
    relative = path.relative_to(scratch)
    parts = relative.parts
    return ((len(parts) == 2 and re.fullmatch(r"input_\d{4}", parts[0])
             and parts[1] == "inputs.pt") or
            (len(parts) == 4 and re.fullmatch(r"forward_\d{4}", parts[0])
             and parts[1] == "windows" and re.fullmatch(r"\d{4}", parts[2])
             and parts[3] == "local.npz") or
            (len(parts) == 2 and re.fullmatch(r"score_\d{4}", parts[0])
             and parts[1] == "pointcloud.tmp"))


def cleanup_regenerable(scratch, output, run_id):
    """Delete only owned, recorded temporary files after complete scene scoring."""
    scratch, output = Path(scratch).absolute(), Path(output).absolute()
    owner = read_json(scratch / "owner.json")
    if owner != dict(output=str(output.resolve()), run_id=run_id):
        raise ValueError("scratch ownership mismatch")
    _validated_final(output)
    receipt_path = output / "cleanup_receipt.json"
    if receipt_path.is_file():
        receipt = read_json(receipt_path)
        if receipt.get("run_id") != run_id:
            raise ValueError("cleanup receipt run ID mismatch")
        for item in receipt["deleted"]:
            path = _within_and_no_symlink(scratch, item["path"])
            if path.exists():
                raise ValueError(f"cleanup receipt lists a live file: {path}")
        return receipt
    if scratch.is_symlink() or output.is_symlink():
        raise ValueError("symlink run directory")
    for path in scratch.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"symlink inside owned scratch: {path}")
    plan_path = output / "cleanup_plan.json"
    if plan_path.is_file():
        plan = read_json(plan_path)
    else:
        plan = []
        for path in sorted(scratch.rglob("*")):
            if not path.is_file() or not _regenerable(path, scratch):
                continue
            _within_and_no_symlink(scratch, path)
            info = path.stat()
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise ValueError(f"not a uniquely owned regular file: {path}")
            plan.append(dict(path=str(path), bytes=info.st_size, sha256=sha256(path)))
        atomic_json(plan_path, dict(run_id=run_id, files=plan))
    if isinstance(plan, dict):
        if plan.get("run_id") != run_id:
            raise ValueError("cleanup plan run ID mismatch")
        plan = plan["files"]
    deleted = []
    for item in plan:
        path = _within_and_no_symlink(scratch, item["path"])
        if not _regenerable(path, scratch):
            raise ValueError(f"cleanup path is not allowlisted: {path}")
        if path.exists():
            info = path.stat()
            if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or
                    info.st_size != item["bytes"] or sha256(path) != item["sha256"]):
                raise ValueError(f"cleanup target changed: {path}")
            path.unlink()
        deleted.append(item)
    receipt = dict(run_id=run_id, deleted=deleted,
                   deleted_bytes=sum(item["bytes"] for item in deleted),
                   scope="owned scratch inputs.pt, local.npz and pointcloud.tmp only")
    atomic_json(receipt_path, receipt)
    return receipt


def aggregate_completed(output_root, budget, scene_ids):
    """Partial summaries never treat missing or failed scenes as measured scores."""
    if budget not in BUDGETS:
        raise ValueError("unsupported ScanNet frame budget")
    successful = []
    missing = []
    for scene_id in scene_ids:
        if not SCENE_PATTERN.fullmatch(scene_id):
            raise ValueError("invalid scene ID")
        path = Path(output_root) / f"f{budget}" / scene_id
        if not (path / "COMPLETE.json").is_file():
            missing.append(scene_id)
            continue
        try:
            metrics = _validated_final(path)
        except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError):
            missing.append(scene_id)
            continue
        successful.append((scene_id, metrics))
    averages = {key: float(np.mean([metrics[key] for _, metrics in successful]))
                for key in METRIC_KEYS} if successful else {}
    return dict(complete=not missing and len(successful) == len(scene_ids),
                expected_count=len(scene_ids), success_count=len(successful),
                successful_scenes=[scene for scene, _ in successful],
                missing_scenes=missing, average_metrics=averages,
                protocol_id="fastvggt_scannet_evo132")
