"""Prepare and validate immutable ScanNet evaluation inputs."""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
import tempfile
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, UnidentifiedImageError

from .sens import SensReader, sample_frame_ids, select_frame_ids


_SCHEMA_VERSION = 1
_SCENE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class DatasetValidationError(ValueError):
    """Raised when a prepared scene is incomplete or corrupt."""


@dataclass(frozen=True)
class SceneData:
    scene_id: str
    frame_ids: tuple[int, ...]
    image_paths: tuple[Path, ...]
    poses_c2w: np.ndarray
    intrinsics_color: np.ndarray
    gt_ply: Path
    manifest_sha256: str


def _validate_scene_id(scene_id: str) -> str:
    if (
        not isinstance(scene_id, str)
        or not _SCENE_ID.fullmatch(scene_id)
        or scene_id in {".", ".."}
        or ".." in scene_id
    ):
        raise ValueError(f"invalid scene id: {scene_id!r}")
    return scene_id


def _normalize_scene_ids(scene_ids: Iterable[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for scene_id in scene_ids:
        scene_id = _validate_scene_id(scene_id)
        if scene_id in seen:
            raise ValueError(f"duplicate scene id: {scene_id}")
        seen.add(scene_id)
        normalized.append(scene_id)
    if not normalized:
        raise ValueError("at least one scene id is required")
    return normalized


def read_scene_list(path: str | os.PathLike[str]) -> list[str]:
    scene_ids = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        value = line.split("#", 1)[0].strip()
        if value:
            scene_ids.append(value)
    return _normalize_scene_ids(scene_ids)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_matrix(path: Path, matrix: np.ndarray) -> None:
    np.savetxt(path, np.asarray(matrix), fmt="%.9g")


def _verify_image_bytes(payload: bytes, expected_size: tuple[int, int], frame_id: int) -> None:
    try:
        with Image.open(io.BytesIO(payload)) as image:
            actual_size = image.size
            image.verify()
        with Image.open(io.BytesIO(payload)) as image:
            image.load()
    except (OSError, UnidentifiedImageError) as error:
        raise DatasetValidationError(f"frame {frame_id} image failed to decode") from error
    if actual_size != expected_size:
        raise DatasetValidationError(
            f"frame {frame_id} image size {actual_size} does not match sensor size {expected_size}"
        )


def _source_paths(raw_root: Path, scene_id: str) -> tuple[Path, Path]:
    sens = raw_root / "raw_sens" / "scans" / scene_id / f"{scene_id}.sens"
    gt = raw_root / "raw" / "scans" / scene_id / f"{scene_id}_vh_clean_2.ply"
    if not sens.is_file():
        raise FileNotFoundError(f"missing raw .sens file: {sens}")
    if not gt.is_file():
        raise FileNotFoundError(f"missing raw GT PLY: {gt}")
    return sens, gt


def _paths_overlap(first: Path, second: Path) -> bool:
    return first == second or first in second.parents or second in first.parents


def _preflight_output_destinations(
    raw_root: Path, output_root: Path, scene_ids: Iterable[str]
) -> None:
    """Reject any output scene that resolves inside or around a raw source tree."""

    scene_ids = tuple(scene_ids)
    resolved_sources: list[tuple[str, Path]] = []
    for scene_id in scene_ids:
        sens_path, gt_path = _source_paths(raw_root, scene_id)
        resolved_sources.extend(
            (
                (scene_id, sens_path.resolve(strict=True)),
                (scene_id, sens_path.parent.resolve(strict=True)),
                (scene_id, gt_path.resolve(strict=True)),
                (scene_id, gt_path.parent.resolve(strict=True)),
            )
        )
    for scene_id in scene_ids:
        destination = (output_root / scene_id).resolve(strict=False)
        for source_scene_id, source in resolved_sources:
            if _paths_overlap(destination, source):
                raise ValueError(
                    f"prepared output scene would overlap immutable raw data for "
                    f"{scene_id}: destination={destination}, "
                    f"source_scene={source_scene_id}, source={source}"
                )


def _source_signature(sens: Path, gt: Path) -> dict[str, object]:
    sens_stat = sens.stat()
    gt_stat = gt.stat()
    return {
        "sens_size": sens_stat.st_size,
        "sens_mtime_ns": sens_stat.st_mtime_ns,
        "gt_size": gt_stat.st_size,
        "gt_sha256": _sha256(gt),
    }


def _read_manifest(scene_dir: Path) -> tuple[dict[str, object], Path]:
    path = scene_dir / "manifest.json"
    if not path.is_file():
        raise DatasetValidationError(f"missing manifest: {path}")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DatasetValidationError(f"invalid manifest: {path}") from error
    if not isinstance(manifest, dict):
        raise DatasetValidationError(f"invalid manifest object: {path}")
    return manifest, path


def _expected_relative_files(scene_id: str, frame_ids: tuple[int, ...], extension: str) -> set[str]:
    result = {
        "calibration/intrinsics_color.txt",
        "calibration/extrinsics_color.txt",
        "calibration/intrinsics_depth.txt",
        "calibration/extrinsics_depth.txt",
        "calibration/sensor.json",
        f"gt/{scene_id}_vh_clean_2.ply",
    }
    result.update(f"color/{frame_id:06d}{extension}" for frame_id in frame_ids)
    result.update(f"pose/{frame_id:06d}.txt" for frame_id in frame_ids)
    return result


def _validate_scene(
    scene_dir: Path,
    scene_id: str,
    verify_hashes: bool,
    *,
    max_frames: int | None = None,
) -> SceneData:
    manifest, manifest_path = _read_manifest(scene_dir)
    if manifest.get("schema_version") != _SCHEMA_VERSION or manifest.get("scene_id") != scene_id:
        raise DatasetValidationError(f"manifest identity/schema mismatch for {scene_id}")
    raw_ids = manifest.get("frame_ids")
    if (
        not isinstance(raw_ids, list)
        or not raw_ids
        or any(isinstance(value, bool) or not isinstance(value, int) for value in raw_ids)
    ):
        raise DatasetValidationError(f"invalid or empty frame_ids for {scene_id}")
    frame_ids = tuple(raw_ids)
    if tuple(sorted(set(frame_ids))) != frame_ids:
        raise DatasetValidationError(f"frame_ids must be strictly increasing for {scene_id}")
    loaded_frame_ids = (
        sample_frame_ids(frame_ids, max_frames=max_frames)
        if max_frames is not None
        else frame_ids
    )
    selection = manifest.get("frame_selection")
    if not isinstance(selection, dict) or selection.get("policy") != "fastvggt_build_frame_selection":
        raise DatasetValidationError(f"invalid frame selection policy for {scene_id}")
    preparation_cap = selection.get("preparation_cap")
    if preparation_cap is not None and (
        isinstance(preparation_cap, bool) or not isinstance(preparation_cap, int) or preparation_cap <= 0
    ):
        raise DatasetValidationError(f"invalid preparation cap for {scene_id}")
    if preparation_cap is not None and len(frame_ids) > preparation_cap:
        raise DatasetValidationError(f"scene {scene_id} exceeds preparation cap={preparation_cap}")
    extension = manifest.get("color_extension")
    if extension not in {".jpg", ".png"}:
        raise DatasetValidationError(f"invalid color extension for {scene_id}")
    color_size_value = manifest.get("color_size")
    if (
        not isinstance(color_size_value, list)
        or len(color_size_value) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in color_size_value)
    ):
        raise DatasetValidationError(f"invalid color size for {scene_id}")
    color_size = (color_size_value[0], color_size_value[1])

    hashes = manifest.get("files")
    if not isinstance(hashes, dict) or any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in hashes.items()
    ):
        raise DatasetValidationError(f"invalid file hash table for {scene_id}")
    expected = _expected_relative_files(scene_id, frame_ids, extension)
    if set(hashes) != expected:
        missing = sorted(expected - set(hashes))
        extra = sorted(set(hashes) - expected)
        raise DatasetValidationError(f"incomplete file hash table for {scene_id}; missing={missing}, extra={extra}")
    actual = {
        path.relative_to(scene_dir).as_posix()
        for path in scene_dir.rglob("*")
        if path.is_file() and path != manifest_path
    }
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise DatasetValidationError(f"missing or unexpected files for {scene_id}; missing={missing}, extra={extra}")

    for relative in sorted(expected):
        path = scene_dir / relative
        if not path.is_file():
            raise DatasetValidationError(f"missing file for {scene_id}: {relative}")
        if verify_hashes and _sha256(path) != hashes[relative]:
            raise DatasetValidationError(f"hash mismatch for {scene_id}: {relative}")

    image_paths = tuple(
        scene_dir / "color" / f"{frame_id:06d}{extension}"
        for frame_id in loaded_frame_ids
    )
    for frame_id, image_path in zip(loaded_frame_ids, image_paths, strict=True):
        _verify_image_bytes(image_path.read_bytes(), color_size, frame_id)
    poses = []
    for frame_id in loaded_frame_ids:
        pose_path = scene_dir / "pose" / f"{frame_id:06d}.txt"
        try:
            pose = np.loadtxt(pose_path, dtype=np.float64)
        except (OSError, ValueError) as error:
            raise DatasetValidationError(f"invalid pose TXT for frame {frame_id}") from error
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            raise DatasetValidationError(f"invalid non-finite pose for frame {frame_id}")
        poses.append(pose)
    intrinsic_path = scene_dir / "calibration" / "intrinsics_color.txt"
    try:
        intrinsic_matrix = np.loadtxt(intrinsic_path, dtype=np.float64)
    except (OSError, ValueError) as error:
        raise DatasetValidationError(f"invalid color intrinsics for {scene_id}") from error
    if intrinsic_matrix.shape != (4, 4) or not np.isfinite(intrinsic_matrix).all():
        raise DatasetValidationError(f"invalid color intrinsics for {scene_id}")
    intrinsics = intrinsic_matrix[:3, :3].copy()
    for calibration_name in ("extrinsics_color", "intrinsics_depth", "extrinsics_depth"):
        try:
            matrix = np.loadtxt(scene_dir / "calibration" / f"{calibration_name}.txt", dtype=np.float64)
        except (OSError, ValueError) as error:
            raise DatasetValidationError(f"invalid {calibration_name} for {scene_id}") from error
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            raise DatasetValidationError(f"invalid {calibration_name} for {scene_id}")
    sensor_path = scene_dir / "calibration" / "sensor.json"
    try:
        sensor_metadata = json.loads(sensor_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DatasetValidationError(f"invalid sensor metadata for {scene_id}") from error
    if (
        not isinstance(sensor_metadata, dict)
        or sensor_metadata.get("color_size") != list(color_size)
        or not isinstance(sensor_metadata.get("depth_size"), list)
        or len(sensor_metadata["depth_size"]) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in sensor_metadata["depth_size"]
        )
        or not isinstance(sensor_metadata.get("color_compression"), int)
        or not isinstance(sensor_metadata.get("depth_compression"), int)
        or not isinstance(sensor_metadata.get("depth_shift"), (int, float))
        or not np.isfinite(sensor_metadata["depth_shift"])
        or sensor_metadata["depth_shift"] <= 0
    ):
        raise DatasetValidationError(f"invalid sensor metadata for {scene_id}")
    gt_ply = scene_dir / "gt" / f"{scene_id}_vh_clean_2.ply"
    with gt_ply.open("rb") as stream:
        if stream.read(3) != b"ply":
            raise DatasetValidationError(f"invalid GT PLY for {scene_id}")
    return SceneData(
        scene_id=scene_id,
        frame_ids=loaded_frame_ids,
        image_paths=image_paths,
        poses_c2w=np.stack(poses),
        intrinsics_color=intrinsics,
        gt_ply=gt_ply,
        manifest_sha256=_sha256(manifest_path),
    )


def validate_dataset(
    prepared_root: str | os.PathLike[str],
    scene_ids: Iterable[str] | None = None,
    verify_hashes: bool = True,
) -> dict[str, object]:
    root = Path(prepared_root)
    if scene_ids is None:
        if not root.is_dir():
            raise DatasetValidationError(f"prepared root does not exist: {root}")
        selected = sorted(path.name for path in root.iterdir() if path.is_dir() and not path.name.startswith("."))
        scene_list = _normalize_scene_ids(selected)
    else:
        scene_list = _normalize_scene_ids(scene_ids)
    for scene_id in scene_list:
        scene_dir = root / scene_id
        if not scene_dir.is_dir():
            raise DatasetValidationError(f"missing prepared scene directory: {scene_dir}")
        _validate_scene(scene_dir, scene_id, verify_hashes=verify_hashes)
    return {
        "requested_scenes": len(scene_list),
        "valid_scenes": len(scene_list),
        "scene_ids": scene_list,
        "verify_hashes": bool(verify_hashes),
    }


def load_scene(
    prepared_root: str | os.PathLike[str],
    scene_id: str,
    max_frames: int = 1000,
    *,
    verify_hashes: bool = False,
) -> SceneData:
    scene_id = _validate_scene_id(scene_id)
    if isinstance(max_frames, bool) or not isinstance(max_frames, int) or max_frames <= 0:
        raise ValueError("max_frames must be a positive integer")
    scene_dir = Path(prepared_root) / scene_id
    if not scene_dir.is_dir():
        raise DatasetValidationError(f"missing prepared scene directory: {scene_dir}")
    return _validate_scene(
        scene_dir,
        scene_id,
        verify_hashes=bool(verify_hashes),
        max_frames=max_frames,
    )


def _prepare_scene(raw_root: Path, output_root: Path, scene_id: str, max_frames: int | None) -> str:
    sens_path, gt_path = _source_paths(raw_root, scene_id)
    signature = _source_signature(sens_path, gt_path)
    destination = output_root / scene_id
    if destination.is_dir():
        try:
            _validate_scene(destination, scene_id, verify_hashes=True)
            manifest, _ = _read_manifest(destination)
            selection = manifest.get("frame_selection")
            if (
                isinstance(selection, dict)
                and selection.get("preparation_cap") == max_frames
                and manifest.get("source") == signature
            ):
                return "reused"
        except (DatasetValidationError, OSError, ValueError):
            pass

    reader = SensReader(sens_path)
    selected_ids = select_frame_ids(reader.frames, max_frames=max_frames)
    if not selected_ids:
        raise DatasetValidationError(f"scene {scene_id} contains no finite camera poses")
    frame_by_id = {frame.frame_id: frame for frame in reader.frames}
    staging_root = Path(tempfile.mkdtemp(prefix=f".{scene_id}-", suffix=".partial", dir=output_root))
    staging = staging_root / scene_id
    backup: Path | None = None
    try:
        images_dir = staging / "color"
        poses_dir = staging / "pose"
        calibration_dir = staging / "calibration"
        gt_dir = staging / "gt"
        for directory in (images_dir, poses_dir, calibration_dir, gt_dir):
            directory.mkdir(parents=True, exist_ok=True)
        selected_frames = tuple(frame_by_id[frame_id] for frame_id in selected_ids)
        for frame, payload in zip(
            selected_frames, reader.iter_color_payloads(selected_frames), strict=True
        ):
            frame_id = frame.frame_id
            _verify_image_bytes(payload, reader.header.color_size, frame_id)
            (images_dir / f"{frame_id:06d}{reader.header.color_extension}").write_bytes(payload)
            _write_matrix(poses_dir / f"{frame_id:06d}.txt", frame.camera_to_world)

        _write_matrix(calibration_dir / "intrinsics_color.txt", reader.header.intrinsic_color)
        _write_matrix(calibration_dir / "extrinsics_color.txt", reader.header.extrinsic_color)
        _write_matrix(calibration_dir / "intrinsics_depth.txt", reader.header.intrinsic_depth)
        _write_matrix(calibration_dir / "extrinsics_depth.txt", reader.header.extrinsic_depth)
        sensor_metadata = {
            "sensor_name": reader.header.sensor_name,
            "color_compression": reader.header.color_compression,
            "depth_compression": reader.header.depth_compression,
            "color_size": list(reader.header.color_size),
            "depth_size": [reader.header.depth_width, reader.header.depth_height],
            "depth_shift": reader.header.depth_shift,
        }
        (calibration_dir / "sensor.json").write_text(
            json.dumps(sensor_metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        shutil.copyfile(gt_path, gt_dir / gt_path.name)
        relative_files = sorted(
            path.relative_to(staging).as_posix() for path in staging.rglob("*") if path.is_file()
        )
        manifest = {
            "schema_version": _SCHEMA_VERSION,
            "scene_id": scene_id,
            "frame_ids": list(selected_ids),
            "frame_selection": {
                "policy": "fastvggt_build_frame_selection",
                "preparation_cap": max_frames,
            },
            "color_extension": reader.header.color_extension,
            "color_size": list(reader.header.color_size),
            "source": signature,
            "files": {relative: _sha256(staging / relative) for relative in relative_files},
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        _validate_scene(staging, scene_id, verify_hashes=True)

        if destination.exists():
            backup = output_root / f".{scene_id}-{uuid.uuid4().hex}.backup"
            destination.rename(backup)
        try:
            staging.rename(destination)
        except BaseException:
            if backup is not None and backup.exists() and not destination.exists():
                backup.rename(destination)
            raise
        if backup is not None:
            shutil.rmtree(backup)
        return "prepared"
    finally:
        if staging_root.exists():
            shutil.rmtree(staging_root)


def prepare_dataset(
    raw_root: str | os.PathLike[str],
    output_root: str | os.PathLike[str],
    scene_ids: Iterable[str],
    max_frames: int | None = None,
    workers: int = 4,
) -> dict[str, object]:
    scene_list = _normalize_scene_ids(scene_ids)
    if max_frames is not None and (
        isinstance(max_frames, bool) or not isinstance(max_frames, int) or max_frames <= 0
    ):
        raise ValueError("max_frames must be None or a positive integer")
    if isinstance(workers, bool) or not isinstance(workers, int) or workers <= 0:
        raise ValueError("workers must be a positive integer")
    raw_path = Path(raw_root)
    output_path = Path(output_root)
    _preflight_output_destinations(raw_path, output_path, scene_list)
    output_path.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=min(workers, len(scene_list))) as pool:
        statuses = list(
            pool.map(
                lambda scene: _prepare_scene(raw_path, output_path, scene, max_frames),
                scene_list,
            )
        )
    return {
        "requested_scenes": len(scene_list),
        "prepared": statuses.count("prepared"),
        "reused": statuses.count("reused"),
        "scene_ids": scene_list,
        "preparation_cap": max_frames,
        "frame_selection_policy": "fastvggt_build_frame_selection",
    }


__all__ = [
    "DatasetValidationError",
    "SceneData",
    "load_scene",
    "prepare_dataset",
    "read_scene_list",
    "validate_dataset",
]
