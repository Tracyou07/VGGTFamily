"""Version-gated raw RGB/extrinsics and immutable normalized preparation.

Only the NAVER 1.3.1 monocular layout is accepted. Official extrinsics map
world to camera; inversion yields the camera centers used for ATE.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import re
import shutil
import tempfile
import numpy as np
from PIL import Image
from .config import DatasetValidationError, VirtualKittiConfig, PROTOCOL_ID, validate_sequence_ids
from .io import canonical_json, content_sha256, file_record, read_json, sha256_file

SOURCE_URL = "https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-1/"
INTRINSICS = ((725., 0., 620.5), (0., 725., 187.), (0., 0., 1.))

@dataclass(frozen=True)
class Blocker:
    code: str
    message: str

@dataclass(frozen=True)
class PreparedSequence:
    sequence: str
    frame_ids: tuple[str, ...]
    image_paths: tuple[Path, ...]
    timestamps_s: None
    poses_c2w: np.ndarray
    intrinsics: np.ndarray
    manifest_path: Path
    manifest_sha256: str

@dataclass(frozen=True)
class BackendRequest:
    frame_ids: tuple[str, ...]
    image_paths: tuple[Path, ...]

    @property
    def input_fields(self):
        return ("frame_ids", "image_paths")

def make_backend_request(prepared: PreparedSequence) -> BackendRequest:
    return BackendRequest(tuple(prepared.frame_ids), tuple(prepared.image_paths))

@dataclass(frozen=True)
class _RawSequence:
    frame_ids: tuple[str, ...]
    image_paths: tuple[Path, ...]
    poses_c2w: np.ndarray
    sources: tuple[dict, ...]

@dataclass(frozen=True)
class DatasetStatus:
    sequence: str
    ready: bool
    blockers: tuple[Blocker, ...]
    _raw: _RawSequence | None = field(default=None, repr=False, compare=False)

def _error(code, message):
    raise DatasetValidationError(code, message)

def _readonly(value):
    value = np.asarray(value, dtype=np.float64)
    return np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)

def _check_sequence(config, sequence):
    validate_sequence_ids(config.sequence_ids)
    if sequence not in config.sequence_ids:
        _error("INVALID_SEQUENCE", f"{sequence!r} is not explicitly configured")
    if config.dataset_version != "1.3.1":
        _error("DATASET_VERSION_MISMATCH", "Formal version must be 1.3.1")
    if config.camera != "monocular":
        _error("INVALID_CAMERA", "Only the 1.3.1 monocular camera is supported")
    for source in (config.raw_root.resolve(), config.archive_root.resolve()):
        target = config.prepared_root.resolve()
        if target.is_relative_to(source) or source.is_relative_to(target):
            _error("INVALID_CONFIG", "Prepared output must not overlap source data")

def read_dataset_version(raw_root: Path) -> str | None:
    """Versioned official top-level names and optional VERSION are evidence.

    A VERSION file alone proves no layout. Conflicting evidence is rejected.
    This is a filesystem version marker, not a claim of archive authenticity.
    """
    versions = set()
    marker = raw_root / "VERSION"
    if marker.is_file():
        versions.add(marker.read_text().strip())
    if raw_root.is_dir():
        for child in raw_root.iterdir():
            match = re.fullmatch(r"vkitti_([0-9]+\.[0-9]+\.[0-9]+)_(rgb|extrinsicsgt|textgt)", child.name)
            if match and child.is_dir():
                versions.add(match[1])
    if len(versions) == 1:
        return next(iter(versions))
    return "conflicting:" + ",".join(sorted(versions)) if versions else None

def _paths(config, sequence):
    scene, condition = sequence.split("/")
    world = f"{int(scene[5:]):04d}"
    image_root = config.raw_root / "vkitti_1.3.1_rgb" / world / condition.lower()
    pose_file = config.raw_root / "vkitti_1.3.1_extrinsicsgt" / f"{world}_{condition.lower()}.txt"
    return image_root, pose_file

def _validate_poses(poses, count):
    if poses.shape != (count, 4, 4) or not np.isfinite(poses).all():
        _error("INVALID_POSES", "Expected finite N x 4 x 4 poses")
    rotations = poses[:, :3, :3]
    if (not np.allclose(poses[:, 3], [0,0,0,1], atol=1e-7, rtol=0)
        or not np.allclose(rotations @ rotations.transpose(0,2,1), np.eye(3), atol=1e-5, rtol=0)
        or not np.allclose(np.linalg.det(rotations), 1, atol=1e-5, rtol=0)):
        _error("INVALID_POSES", "Expected right-handed rigid transforms")

def _parse_raw(config, images, pose_file):
    ids = tuple(p.stem for p in images)
    if ids != tuple(f"{i:05d}" for i in range(len(ids))):
        _error("INVALID_FRAME_ID", "RGB requires contiguous %05d.png IDs starting at zero")
    sources = [*images, pose_file]
    marker = config.raw_root / "VERSION"
    if marker.is_file():
        sources.append(marker)
    records = tuple(file_record(p, config.raw_root) for p in sources)
    size = None
    for path in images:
        try:
            with Image.open(path) as im:
                if im.format != "PNG":
                    raise ValueError("Only official RGB PNGs are supported")
                im.verify()
            with Image.open(path) as im:
                im.load()
                if im.mode != "RGB":
                    raise ValueError("Expected RGB pixels")
                if size is None:
                    size = im.size
                if size != im.size:
                    raise ValueError("Inconsistent image dimensions")
        except (OSError, ValueError, SyntaxError) as exc:
            raise DatasetValidationError("INVALID_IMAGE", f"{path}: {exc}") from exc
    try:
        lines = [line.split() for line in pose_file.read_text().splitlines() if line.strip()]
        # Official CSV-like file has a header, frame index, and 16 row-major values.
        # Matrix column labels are descriptive; their positions are authoritative.
        if not lines or len(lines[0]) != 17 or lines[0][0] != "frame":
            raise ValueError("Expected frame + 16-column extrinsics header")
        if any(len(row) != 17 for row in lines[1:]):
            raise ValueError("1.3.1 extrinsics have no camera-ID column")
        gt_ids = tuple(row[0] for row in lines[1:])
        if len(gt_ids) != len(ids):
            _error("COUNT_MISMATCH", "Image and pose counts differ")
        if gt_ids != tuple(str(i) for i in range(len(ids))):
            _error("INVALID_FRAME_ID", "GT rows must match exact RGB order from zero")
        poses = np.asarray([[float(v) for v in row[1:]] for row in lines[1:]], dtype=float).reshape(-1,4,4)
        _validate_poses(poses, len(ids))
        with np.errstate(over="raise", invalid="raise"):
            c2w = np.repeat(np.eye(4)[None], len(poses), axis=0)
            c2w[:, :3, :3] = poses[:, :3, :3].transpose(0,2,1)
            c2w[:, :3, 3] = -np.einsum("nij,nj->ni", c2w[:, :3, :3], poses[:, :3, 3])
        _validate_poses(c2w, len(ids))
    except DatasetValidationError:
        raise
    except (OSError, ValueError, FloatingPointError) as exc:
        raise DatasetValidationError("INVALID_POSES", str(exc)) from exc
    if records != tuple(file_record(p, config.raw_root) for p in sources):
        _error("SOURCE_CHANGED", "Raw inputs changed during validation")
    return _RawSequence(ids, tuple(images), _readonly(c2w), records)

def inspect_raw_sequence(config: VirtualKittiConfig, sequence: str) -> DatasetStatus:
    blockers = []
    try:
        _check_sequence(config, sequence)
        version = read_dataset_version(config.raw_root)
        if version != "1.3.1":
            blockers.append(Blocker("DATASET_VERSION_MISMATCH", f"expected 1.3.1, found {version}"))
        if any(p.is_file() for p in config.archive_root.rglob("*rgb*.part")):
            blockers.append(Blocker("INCOMPLETE_RGB_ARCHIVE", "partial RGB archive present"))
        # Version and incomplete archive gate occurs before any layout discovery.
        if blockers:
            return DatasetStatus(sequence, False, tuple(blockers))
        if not any((config.raw_root / name).is_dir() for name in ("vkitti_1.3.1_rgb", "vkitti_1.3.1_extrinsicsgt")):
            return DatasetStatus(sequence, False, (Blocker("DATASET_LAYOUT_UNVERIFIED", SOURCE_URL),))
        image_root, pose_file = _paths(config, sequence)
        images = sorted(image_root.iterdir()) if image_root.is_dir() else []
        if not images:
            blockers.append(Blocker("MISSING_RGB", str(image_root)))
        elif any(not p.is_file() or re.fullmatch(r"[0-9]{5}\.png", p.name) is None for p in images):
            blockers.append(Blocker("INVALID_FRAME_ID", "Only official %05d.png frames are accepted"))
        if not pose_file.is_file():
            blockers.append(Blocker("MISSING_POSES", str(pose_file)))
        if blockers:
            return DatasetStatus(sequence, False, tuple(blockers))
        raw = _parse_raw(config, images, pose_file)
        return DatasetStatus(sequence, True, (), raw)
    except DatasetValidationError as exc:
        blockers.append(Blocker(exc.code, exc.message))
    except (OSError, ValueError, TypeError) as exc:
        blockers.append(Blocker("RAW_IO", str(exc)))
    return DatasetStatus(sequence, False, tuple(blockers))

def _manifest(sequence, raw, target, config):
    value = {"schema_version": 1, "protocol_id": PROTOCOL_ID, "dataset_version": "1.3.1",
        "layout_source": SOURCE_URL, "camera": "monocular", "sequence": sequence,
        "frame_ids": list(raw.frame_ids),
        "image_paths": [p.relative_to(config.raw_root).as_posix() for p in raw.image_paths],
        "timestamps_s": None, "intrinsics": [list(row) for row in INTRINSICS],
        "sources": list(raw.sources), "prepared_files": [file_record(target / "poses_c2w.npy", target)]}
    value["content_sha256"] = content_sha256(value)
    return value

def _target(config, sequence):
    target = config.prepared_root / sequence
    if not target.resolve().is_relative_to(config.prepared_root.resolve()):
        _error("INVALID_MANIFEST", "Prepared path escapes prepared_root")
    return target

def prepare_sequence(config: VirtualKittiConfig, sequence: str) -> PreparedSequence:
    status = inspect_raw_sequence(config, sequence)
    if not status.ready:
        _error(status.blockers[0].code, status.blockers[0].message)
    target = _target(config, sequence)
    if target.exists():
        return verify_prepared_sequence(config, sequence)
    temporary = None
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}-", dir=target.parent))
        np.save(temporary / "poses_c2w.npy", status._raw.poses_c2w, allow_pickle=False)
        (temporary / "manifest.json").write_bytes(canonical_json(_manifest(sequence, status._raw, temporary, config)) + b"\n")
        if status._raw.sources != tuple(file_record(config.raw_root / r["path"], config.raw_root) for r in status._raw.sources):
            _error("SOURCE_CHANGED", "Raw inputs changed before publication")
        temporary.rename(target)
        temporary = None
    except DatasetValidationError:
        raise
    except (OSError, ValueError) as exc:
        raise DatasetValidationError("PREPARE_IO", str(exc)) from exc
    finally:
        if temporary is not None:
            shutil.rmtree(temporary)
    return verify_prepared_sequence(config, sequence)

def verify_prepared_sequence(config: VirtualKittiConfig, sequence: str, verify_hashes: bool = True) -> PreparedSequence:
    """Always verify full source content, even for callers requesting a fast check."""
    status = inspect_raw_sequence(config, sequence)
    if not status.ready:
        _error(status.blockers[0].code, status.blockers[0].message)
    target = _target(config, sequence)
    manifest_path = target / "manifest.json"
    try:
        if not manifest_path.resolve().is_relative_to(target.resolve()):
            _error("INVALID_MANIFEST", "Manifest escapes prepared directory")
        manifest = read_json(manifest_path)
        if canonical_json(manifest) != canonical_json(_manifest(sequence, status._raw, target, config)):
            _error("MANIFEST_OR_SOURCE_CHANGED", "Manifest does not match current validated sources")
        pose_path = target / "poses_c2w.npy"
        if not pose_path.resolve().is_relative_to(target.resolve()):
            _error("INVALID_MANIFEST", "Normalized poses escape prepared directory")
        with pose_path.open("rb") as stream:
            if stream.read(6) != b"\x93NUMPY":
                _error("PREPARED_CHANGED", "Normalized poses require a standalone NPY array")
        poses = np.load(pose_path, allow_pickle=False)
        if poses.dtype.kind != "f" or not np.array_equal(poses, status._raw.poses_c2w):
            _error("PREPARED_CHANGED", "Normalized c2w differs from official extrinsics")
        return PreparedSequence(sequence, status._raw.frame_ids, status._raw.image_paths, None,
            _readonly(poses), _readonly(INTRINSICS), manifest_path, sha256_file(manifest_path))
    except DatasetValidationError:
        raise
    except (OSError, ValueError, TypeError, KeyError, EOFError) as exc:
        raise DatasetValidationError("INVALID_MANIFEST", str(exc)) from exc
