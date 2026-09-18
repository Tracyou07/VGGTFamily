"""Read-only raw discovery and immutable KITTI image_2 preparation.

Official pose rows are camera-0 camera-to-world. Rectified projection
P_i = K_i [I | t_i] gives camera center -t_i in the calibration reference.
Consequently T_cam0_from_cam2 translates by t_0 - t_2, and
T_world_from_cam2 = T_world_from_cam0 @ T_cam0_from_cam2.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import json
from pathlib import Path
import re
import shutil
import tempfile
import numpy as np
from PIL import Image
from .config import DatasetValidationError, KittiConfig, PROTOCOL_ID
from .io import canonical_json, content_sha256, file_record, sha256_file

@dataclass(frozen=True)
class Blocker:
    code: str
    message: str

@dataclass(frozen=True)
class PreparedSequence:
    sequence: str
    frame_ids: tuple[str, ...]
    image_paths: tuple[Path, ...]
    timestamps_s: np.ndarray
    poses_c2w: np.ndarray
    intrinsics: np.ndarray
    manifest_path: Path
    manifest_sha256: str

@dataclass(frozen=True)
class _RawSequence:
    frame_ids: tuple[str, ...]
    image_paths: tuple[Path, ...]
    timestamps_s: np.ndarray
    poses_c2w: np.ndarray
    intrinsics: np.ndarray
    camera0_from_camera2: np.ndarray
    sources: tuple[dict, ...]

@dataclass(frozen=True)
class DatasetStatus:
    sequence: str
    ready: bool
    blockers: tuple[Blocker, ...]
    _raw: _RawSequence | None = field(default=None, repr=False, compare=False)

def _error(code: str, message: str):
    raise DatasetValidationError(code, message)

def _check_sequence(config: KittiConfig, sequence: str):
    if not isinstance(sequence, str) or re.fullmatch(r"[0-9]{2}", sequence) is None or sequence not in config.sequence_ids:
        _error("INVALID_SEQUENCE", f"Sequence {sequence!r} is not in the configured sequence list")

def _readonly(array):
    array = np.asarray(array, dtype=np.float64)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)

def _validate_poses(poses, count, code="INVALID_POSES"):
    if poses.shape != (count, 4, 4) or not np.isfinite(poses).all():
        _error(code, "Poses must be finite N x 4 x 4 matrices")
    if not np.allclose(poses[:, 3], [0, 0, 0, 1], atol=1e-7, rtol=0):
        _error(code, "Invalid homogeneous pose row")
    rotations = poses[:, :3, :3]
    if not np.allclose(rotations.transpose(0, 2, 1) @ rotations, np.eye(3), atol=1e-4, rtol=0) or not np.allclose(np.linalg.det(rotations), 1, atol=1e-4, rtol=0):
        _error(code, "Pose rotations must be rigid and right handed")

def _calibration(path):
    try:
        projections = {}
        for line in path.read_text().splitlines():
            if not line.strip(): continue
            key, values = line.split(":", 1)
            key = key.strip()
            if key not in ("P0", "P2"): continue
            if key in projections: raise ValueError(f"Duplicate {key}")
            p = np.array([float(v) for v in values.split()]).reshape(3, 4)
            k = p[:, :3]
            if not np.isfinite(p).all() or k[0,0] <= 0 or k[1,1] <= 0 or not np.allclose(k[2], [0,0,1], atol=1e-8) or not np.allclose(np.tril(k, -1), 0, atol=1e-8):
                raise ValueError(f"{key} must be a finite rectified projection")
            projections[key] = (k, np.linalg.solve(k, p[:, 3]))
        k0, t0 = projections["P0"]
        k2, t2 = projections["P2"]
        transform = np.eye(4)
        with np.errstate(over="raise", invalid="raise"):
            transform[:3, 3] = t0 - t2
        if not np.isfinite(transform).all():
            raise ValueError("Non-finite camera conversion")
        return k2, transform
    except (OSError, ValueError, KeyError, FloatingPointError, np.linalg.LinAlgError) as exc:
        raise DatasetValidationError("INVALID_CALIBRATION", str(exc)) from exc

def _parse_raw(config, sequence, images, pose_file, calib_file, times_file):
    ids = tuple(p.stem for p in images)
    if len(set(ids)) != len(ids):
        _error("DUPLICATE_FRAME_ID", "Multiple images use the same frame ID")
    if any(re.fullmatch(r"[0-9]{6}", i) is None for i in ids) or ids != tuple(f"{i:06d}" for i in range(len(ids))):
        _error("INVALID_FRAME_ID", "Frames must be contiguous six-digit IDs starting at 000000")
    sources = tuple(file_record(p, config.raw_root) for p in (*images, pose_file, calib_file, times_file))
    size = None
    for path in images:
        try:
            with Image.open(path) as im:
                if im.format not in ("PNG", "JPEG"): raise ValueError("Only PNG/JPEG supported")
                im.verify()
            with Image.open(path) as im:
                im.load()  # verify alone does not decode compressed JPEG pixel data.
                if im.mode != "RGB": raise ValueError("image_2 must contain RGB images")
                if size is None: size = im.size
                if im.size != size: raise ValueError("Inconsistent image dimensions")
        except (OSError, ValueError, SyntaxError) as exc:
            raise DatasetValidationError("INVALID_IMAGE", f"{path}: {exc}") from exc
    try:
        rows = [[float(v) for v in line.split()] for line in pose_file.read_text().splitlines() if line.strip()]
        rows = np.asarray(rows, dtype=np.float64)
        if rows.ndim != 2 or rows.shape[1] != 12: raise ValueError("Expected 12 columns per pose row")
    except (OSError, ValueError) as exc:
        raise DatasetValidationError("INVALID_POSES", str(exc)) from exc
    if len(rows) != len(images): _error("COUNT_MISMATCH", "Image and pose counts differ")
    poses = np.repeat(np.eye(4)[None], len(rows), axis=0)
    poses[:, :3] = rows.reshape(-1, 3, 4)
    _validate_poses(poses, len(images))
    try:
        time_rows = [line.split() for line in times_file.read_text().splitlines() if line.strip()]
        if any(len(row) != 1 for row in time_rows): raise ValueError("Expected one timestamp per row")
        timestamps = np.array([float(row[0]) for row in time_rows])
        if not np.isfinite(timestamps).all() or np.any(timestamps < 0) or np.any(np.diff(timestamps) <= 0):
            raise ValueError("Timestamps must be finite, nonnegative and strictly increasing")
    except (OSError, ValueError) as exc:
        raise DatasetValidationError("INVALID_TIMESTAMPS", str(exc)) from exc
    if len(timestamps) != len(images): _error("COUNT_MISMATCH", "Image and timestamp counts differ")
    intrinsics, camera0_from_camera2 = _calibration(calib_file)
    try:
        with np.errstate(over="raise", invalid="raise"):
            poses = poses @ camera0_from_camera2
    except FloatingPointError as exc:
        raise DatasetValidationError("INVALID_POSES", "Camera conversion overflow") from exc
    _validate_poses(poses, len(images))
    return _RawSequence(ids, tuple(images), _readonly(timestamps), _readonly(poses), _readonly(intrinsics),
                        _readonly(camera0_from_camera2), sources)

def inspect_raw_sequence(config: KittiConfig, sequence: str) -> DatasetStatus:
    blockers = []
    try:
        _check_sequence(config, sequence)
        for path in sorted(config.archive_root.rglob("*.part")):
            if path.is_file():
                blockers.append(Blocker("INCOMPLETE_ARCHIVE", str(path)))
        color_sequence_root = config.color_root / "sequences" / sequence
        aux_sequence_root = config.aux_root / "sequences" / sequence
        image_root = color_sequence_root / "image_2"
        pose_file = config.aux_root / "poses" / f"{sequence}.txt"
        calib_file = aux_sequence_root / "calib.txt"
        times_file = color_sequence_root / "times.txt"
        images = sorted(p for p in image_root.iterdir() if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg")) if image_root.is_dir() else []
        if not images: blockers.append(Blocker("EMPTY_IMAGES", f"No image_2 PNG/JPEG frames: {image_root}"))
        for path, code in ((pose_file, "MISSING_POSES"), (calib_file, "MISSING_CALIBRATION"), (times_file, "MISSING_TIMESTAMPS")):
            if not path.is_file(): blockers.append(Blocker(code, str(path)))
        if blockers: return DatasetStatus(sequence, False, tuple(blockers))
        raw = _parse_raw(config, sequence, images, pose_file, calib_file, times_file)
        return DatasetStatus(sequence, True, (), raw)
    except DatasetValidationError as exc:
        blockers.append(Blocker(exc.code, exc.message))
    except (OSError, ValueError) as exc:
        blockers.append(Blocker("RAW_IO", str(exc)))
    return DatasetStatus(sequence, False, tuple(blockers))

def _source_path(root, relative):
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
        _error("INVALID_MANIFEST", "Source paths must be relative and remain under raw_root")
    path = root / relative
    if not path.resolve().is_relative_to(root.resolve()):
        _error("INVALID_MANIFEST", "Source path escapes raw_root")
    return path

def _check_records(records, root, verify_hashes, code):
    for record in records:
        path = _source_path(root, record["path"])
        if not path.is_file() or path.stat().st_size != record["size_bytes"]:
            _error(code, f"Missing or resized file: {path}")
        if verify_hashes and sha256_file(path) != record["sha256"]:
            _error(code, f"SHA-256 mismatch: {path}")

def _atomic_write_prepared(config, sequence, raw):
    target = config.prepared_root / sequence
    if target.exists(): return
    temporary = None
    try:
        config.prepared_root.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".{sequence}-", dir=config.prepared_root))
        np.save(temporary / "poses_c2w.npy", raw.poses_c2w, allow_pickle=False)
        calibration = {"schema_version": 1, "camera": "image_2", "intrinsics": raw.intrinsics.tolist(),
                       "camera0_from_camera2": raw.camera0_from_camera2.tolist()}
        (temporary / "calibration.json").write_bytes(canonical_json(calibration) + b"\n")
        manifest = {"schema_version": 1, "protocol_id": PROTOCOL_ID, "sequence": sequence,
                    "frame_ids": list(raw.frame_ids), "image_paths": [str(p.relative_to(config.raw_root)) for p in raw.image_paths],
                    "timestamps_s": raw.timestamps_s.tolist(), "sources": list(raw.sources),
                    "prepared_files": [file_record(temporary / name, temporary) for name in ("poses_c2w.npy", "calibration.json")]}
        manifest["content_sha256"] = content_sha256(manifest)
        (temporary / "manifest.json").write_bytes(canonical_json(manifest) + b"\n")
        _check_records(raw.sources, config.raw_root, True, "SOURCE_CHANGED")
        # Same filesystem rename publishes the complete directory without replacing existing data.
        temporary.rename(target)
        temporary = None
    except DatasetValidationError:
        raise
    except (OSError, ValueError) as exc:
        raise DatasetValidationError("PREPARE_IO", str(exc)) from exc
    finally:
        if temporary is not None: shutil.rmtree(temporary)

def prepare_sequence(config: KittiConfig, sequence: str) -> PreparedSequence:
    status = inspect_raw_sequence(config, sequence)
    if not status.ready:
        _error(status.blockers[0].code, status.blockers[0].message)
    _atomic_write_prepared(config, sequence, status._raw)
    return verify_prepared_sequence(config, sequence)

def verify_prepared_sequence(config: KittiConfig, sequence: str, verify_hashes: bool = True) -> PreparedSequence:
    _check_sequence(config, sequence)
    status = inspect_raw_sequence(config, sequence)
    # Preserve SOURCE_CHANGED for a formerly valid image whose bytes are now corrupt.
    # Archive/inventory blockers remain immediate; image errors follow integrity checks.
    if not status.ready and status.blockers[0].code != "INVALID_IMAGE":
        _error(status.blockers[0].code, status.blockers[0].message)
    target = config.prepared_root / sequence
    manifest_path = target / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text())
        expected = {"schema_version", "protocol_id", "sequence", "frame_ids", "image_paths", "timestamps_s", "sources", "prepared_files", "content_sha256"}
        if not isinstance(manifest, dict) or set(manifest) != expected: _error("INVALID_MANIFEST", "Unexpected manifest fields")
        digest = manifest.pop("content_sha256")
        if content_sha256(manifest) != digest: _error("MANIFEST_HASH_MISMATCH", str(manifest_path))
        if type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1 or manifest["protocol_id"] != PROTOCOL_ID or manifest["sequence"] != sequence:
            _error("INVALID_MANIFEST", "Unsupported schema, protocol, or sequence")
        ids = manifest["frame_ids"]
        if not isinstance(ids, list) or not ids or ids != [f"{i:06d}" for i in range(len(ids))]:
            _error("INVALID_MANIFEST", "Frame IDs must be unique, contiguous and ordered")
        relative_images = manifest["image_paths"]
        if not isinstance(relative_images, list) or len(relative_images) != len(ids):
            _error("INVALID_MANIFEST", "Image/frame counts differ")
        paths = tuple(_source_path(config.raw_root, p) for p in relative_images)
        color_prefix = config.color_root.relative_to(config.raw_root)
        aux_prefix = config.aux_root.relative_to(config.raw_root)
        expected_image_parent = color_prefix / "sequences" / sequence / "image_2"
        for frame_id, relative in zip(ids, relative_images):
            p = Path(relative)
            if p.parent != expected_image_parent or p.stem != frame_id or p.suffix.lower() not in (".png", ".jpg", ".jpeg"):
                _error("INVALID_MANIFEST", "Images must map to ordered image_2 frame IDs")
        timestamps = np.asarray(manifest["timestamps_s"], dtype=float)
        if timestamps.shape != (len(ids),) or not np.isfinite(timestamps).all() or np.any(timestamps < 0) or np.any(np.diff(timestamps) <= 0):
            _error("INVALID_MANIFEST", "Invalid timestamps")
        sources = manifest["sources"]
        expected_sources = relative_images + [
            str(aux_prefix / "poses" / f"{sequence}.txt"),
            str(aux_prefix / "sequences" / sequence / "calib.txt"),
            str(color_prefix / "sequences" / sequence / "times.txt"),
        ]
        if not isinstance(sources, list) or [s["path"] for s in sources] != expected_sources:
            _error("INVALID_MANIFEST", "Source records must include all images, poses, calibration and times")
        prepared_files = manifest["prepared_files"]
        if not isinstance(prepared_files, list) or [p["path"] for p in prepared_files] != ["poses_c2w.npy", "calibration.json"]:
            _error("INVALID_MANIFEST", "Invalid normalized file inventory")
        for record in sources + prepared_files:
            if set(record) != {"path", "size_bytes", "sha256"} or type(record["size_bytes"]) is not int or record["size_bytes"] < 0 or not isinstance(record["sha256"], str) or re.fullmatch(r"[0-9a-f]{64}", record["sha256"]) is None:
                _error("INVALID_MANIFEST", "Invalid file integrity record")
        _check_records(sources, config.raw_root, verify_hashes, "SOURCE_CHANGED")
        if not status.ready:
            _error(status.blockers[0].code, status.blockers[0].message)
        current = status._raw
        if (tuple(ids) != current.frame_ids or paths != current.image_paths
            or sources != list(current.sources)):
            _error("SOURCE_CHANGED", "Current raw inventory differs from the immutable prepared manifest")
        # The fresh raw inventory above always binds current source hashes.
        # Normalized output is also always integrity-checked.
        _check_records(prepared_files, target, True, "PREPARED_CHANGED")
        poses = np.load(target / "poses_c2w.npy", allow_pickle=False)
        _validate_poses(poses, len(ids))
        calibration = json.loads((target / "calibration.json").read_text())
        if set(calibration) != {"schema_version", "camera", "intrinsics", "camera0_from_camera2"} or type(calibration["schema_version"]) is not int or calibration["schema_version"] != 1 or calibration["camera"] != "image_2":
            _error("INVALID_CALIBRATION", "Invalid normalized calibration schema")
        intrinsics = np.asarray(calibration["intrinsics"], dtype=float)
        if intrinsics.shape != (3,3) or not np.isfinite(intrinsics).all() or intrinsics[0,0] <= 0 or intrinsics[1,1] <= 0 or not np.allclose(intrinsics[2], [0,0,1]) or not np.allclose(np.tril(intrinsics, -1), 0):
            _error("INVALID_CALIBRATION", "Invalid normalized intrinsics")
        transform = np.asarray(calibration["camera0_from_camera2"], dtype=float)
        _validate_poses(transform[None], 1, "INVALID_CALIBRATION")
        return PreparedSequence(sequence, tuple(ids), paths, _readonly(timestamps), _readonly(poses),
                                _readonly(intrinsics), manifest_path, sha256_file(manifest_path))
    except DatasetValidationError:
        raise
    except (OSError, ValueError, TypeError, KeyError, IndexError, AttributeError) as exc:
        raise DatasetValidationError("INVALID_MANIFEST", str(exc)) from exc
