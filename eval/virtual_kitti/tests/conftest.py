from pathlib import Path
from types import SimpleNamespace
import json
import sys
import numpy as np
from PIL import Image
import pytest

CONDITIONS = ("Clone", "Fog", "Morning", "Overcast", "Rain", "Sunset")
MAIN = tuple(f"Scene{s}/{c}" for s in ("01", "02") for c in CONDITIONS)

def make_fixture(root, sequences=("Scene01/Clone",), version="1.3.1"):
    from virtual_kitti_eval.config import load_config
    raw = root / "raw"
    archives = root / "archives"
    raw.mkdir(parents=True)
    archives.mkdir()
    (raw / "VERSION").write_text(version)
    for sequence in sequences:
        scene, condition = sequence.split("/")
        world = f"{int(scene[5:]):04d}"
        images = raw / f"vkitti_{version}_rgb" / world / condition.lower()
        images.mkdir(parents=True)
        # A non-collinear tetrahedron, with nonidentity camera rotation.
        rotation = np.array([[0., -1, 0], [1, 0, 0], [0, 0, 1]])
        centers = np.array([[0., 0, 0], [1, 0, 0], [0, 2, 0], [0, 0, 3]])
        rows = []
        for i in (3, 0, 2, 1):
            Image.new("RGB", (12, 6), (i, 20, 30)).save(images / f"{i:05d}.png")
        for i, center in enumerate(centers):
            w2c = np.eye(4)
            w2c[:3, :3] = rotation
            w2c[:3, 3] = -rotation @ center
            rows.append(str(i) + " " + " ".join(map(str, w2c.reshape(-1))))
        gt = raw / f"vkitti_{version}_extrinsicsgt" / f"{world}_{condition.lower()}.txt"
        gt.parent.mkdir(exist_ok=True)
        gt.write_text("frame " + " ".join(f"m{i}" for i in range(16)) + "\n" + "\n".join(rows) + "\n")
    sequence_file = root / "sequences.txt"
    sequence_file.write_text("\n".join(sequences) + "\n")
    config_path = root / "config.json"
    config_path.write_text(json.dumps({"schema_version": 1, "dataset_version": "1.3.1",
        "camera": "monocular", "raw_root": "raw", "archive_root": "archives",
        "prepared_root": "prepared", "sequences_file": "sequences.txt"}))
    config = load_config(config_path)
    return SimpleNamespace(config=config, config_path=config_path, root=root,
        raw=raw, archives=archives, gt=gt, images=images, centers=centers)

@pytest.fixture
def vkitti_131_fixture(tmp_path):
    return make_fixture(tmp_path)

@pytest.fixture
def vkitti_result_dir(tmp_path):
    from dataclasses import asdict
    from virtual_kitti_eval.data import prepare_sequence
    from virtual_kitti_eval.provenance import RunPlan, build_provenance
    from virtual_kitti_eval.results import atomic_write_json, metrics_record, write_result_pair
    from virtual_kitti_eval.metrics import AteMetrics, Sim3, PROTOCOL_ID
    fixture = make_fixture(tmp_path, MAIN)
    model_root = tmp_path / "model_source"
    model_root.mkdir()
    (model_root / "model.py").write_text("# model fixture\n")
    repo = tmp_path / "eval_source"
    repo.mkdir()
    (repo / "eval.py").write_text("# eval fixture\n")
    checkpoint = tmp_path / "weights.bin"
    checkpoint.write_bytes(b"fixture weights")
    prepared = tuple(prepare_sequence(fixture.config, s) for s in MAIN)
    config = fixture.config
    payload = {"schema_version": 1, "dataset_version": "1.3.1", "camera": "monocular",
        **{key: str(getattr(config, key)) for key in ("raw_root", "archive_root", "prepared_root", "sequences_file")},
        "sequence_ids": list(MAIN)}
    plan = RunPlan(fixture.config_path, payload, "vggt", {"project_root": str(model_root),
        "checkpoint": str(checkpoint), "interpreter": sys.executable, "use_calibration": False},
        MAIN, prepared, tmp_path / "results", "cpu", 60., ("fixture",), repo, PROTOCOL_ID)
    provenance = build_provenance(plan)
    plan.output_dir.mkdir()
    atomic_write_json(plan.output_dir / "run_manifest.json", provenance)
    for index, item in enumerate(prepared):
        metric = metrics_record(AteMetrics(PROTOCOL_ID, 4, float(index + 1),
            Sim3(1., np.eye(3), np.zeros(3))), item.frame_ids, item.sequence, "vggt", provenance["provenance_id"])
        result = {"schema_version": 1, "model_key": "vggt", "sequence": item.sequence, "status": "success",
            "input_frames": 4, "inference_seconds": .1, "peak_allocated_mib": 1., "peak_reserved_mib": 2.,
            "worker_exit_state": {"returncode": 0, "signal": None},
            "provenance_id": provenance["provenance_id"], "metrics_sha256": None}
        write_result_pair(plan.output_dir / item.sequence, result, metric)
    return SimpleNamespace(**vars(fixture), plan=plan, provenance=provenance, output=plan.output_dir,
        checkpoint=checkpoint, model_root=model_root)
