"""Contract for generated, RGB-registered 7-Scenes evaluation depth."""
from pathlib import Path
import hashlib
import json


PROTOCOL = "simplerecon_7scenes_registered_depth_v1"
DEFAULT_DATA_ROOT = "/data/yjh/share/datasets/7scenes_registered_simplerecon_v1"


def registered_depth_path(root, scene_id, frame_id):
    path = Path(root) / scene_id / f"frame-{frame_id}.depth.proj.png"
    if path.is_symlink():
        raise ValueError(
            f"Expected generated registered depth, found a symlink: {path}. "
            f"Use the prepared dataset at {DEFAULT_DATA_ROOT}."
        )
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing registered depth: {path}. Raw depth is not an RGB-registered "
            f"substitute; prepare {DEFAULT_DATA_ROOT} first."
        )
    return str(path)


def validate_registered_root(root, *, verify_hashes=True, source_data_root=None):
    """Validate the completion seal and every test-frame triplet before GPU work."""
    root = Path(root).resolve()
    source_root = Path(source_data_root).resolve() if source_data_root else root
    marker = root / "registration.json"
    if not marker.is_file():
        raise ValueError(f"Dataset is not prepared: missing registration.json in {root}")
    data = json.loads(marker.read_text())
    if data.get("protocol") != PROTOCOL or data.get("complete") is not True:
        raise ValueError(f"Unsupported or incomplete registration dataset: {root}")
    manifest = root / "frames.jsonl"
    contents = manifest.read_bytes()
    if hashlib.sha256(contents).hexdigest() != data.get("frames_manifest_sha256"):
        raise ValueError("Registration frame manifest checksum mismatch")
    seen = set()
    sequences = set()
    for line in contents.decode().splitlines():
        row = json.loads(line)
        scene_id, frame_id = row["sequence"], row["frame"]
        key = (scene_id, frame_id)
        if key in seen:
            raise ValueError(f"Duplicate registered frame: {key}")
        seen.add(key)
        sequences.add(scene_id)
        path = Path(registered_depth_path(root, scene_id, frame_id))
        if verify_hashes and hashlib.sha256(path.read_bytes()).hexdigest() != row["registered_sha256"]:
            raise ValueError(f"Registered depth checksum mismatch: {path}")
        source_sequence = source_root / scene_id
        for suffix in ("color.png", "pose.txt"):
            if not (source_sequence / f"frame-{frame_id}.{suffix}").is_file():
                raise ValueError(f"Incomplete registered frame triplet: {key}")
    if not seen or len(seen) != data.get("frame_count") or len(sequences) != data.get("sequence_count"):
        raise ValueError("Incomplete registration frame/sequence counts")
    listed_sequences = set()
    for split in source_root.glob("*/TestSplit.txt"):
        for value in split.read_text().splitlines():
            sequence = int("".join(filter(str.isdigit, value)))
            listed_sequences.add(f"{split.parent.name}/seq-{sequence:02}")
    if listed_sequences != sequences:
        raise ValueError("Registration manifest disagrees with TestSplit sequences")
    # The loaders index frames contiguously using their RGB count.
    for sequence in sequences:
        count = len(list((source_root / sequence).glob("frame-*.color.png")))
        expected = {(sequence, f"{i:06}") for i in range(count)}
        if expected != {key for key in seen if key[0] == sequence}:
            raise ValueError(f"Registration manifest disagrees with RGB frames: {sequence}")
    return data


def prepare_evaluation(root, output_dir, *, kf, source_data_root=None):
    """Reject old result directories and record the exact registered GT version."""
    data = validate_registered_root(root, source_data_root=source_data_root)
    output_dir = Path(output_dir)
    if output_dir.is_symlink() or (output_dir.exists() and
            (not output_dir.is_dir() or any(output_dir.iterdir()))):
        raise ValueError(f"Use a fresh empty output directory; existing results in {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    source_root = Path(source_data_root).resolve() if source_data_root else Path(root).resolve()
    record = {"protocol": PROTOCOL, "data_root": str(source_root),
              "kf": kf, "frame_count": data["frame_count"],
              "sequence_count": data["sequence_count"],
              "frames_manifest_sha256": data["frames_manifest_sha256"]}
    if source_data_root is not None:
        record["registered_depth_root"] = str(Path(root).resolve())
    with (output_dir / "input_registration.json").open("x") as handle:
        json.dump(record, handle, indent=2)
    print(f"Validated registered GT: {record['frame_count']} frames; {root}", flush=True)
    return record


def prepare_distributed_evaluation(accelerator, root, output_dir, *, kf):
    """Reserve output once and propagate preflight errors to every worker."""
    from accelerate.utils import broadcast_object_list
    error = [None]
    if accelerator.is_main_process:
        try:
            prepare_evaluation(root, output_dir, kf=kf)
        except Exception as exc:
            error[0] = f"Registered GT preflight failed: {type(exc).__name__}: {exc}"
    broadcast_object_list(error)
    if error[0] is not None:
        raise ValueError(error[0])
