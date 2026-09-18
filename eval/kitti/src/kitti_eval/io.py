"""Deterministic JSON and streaming integrity primitives."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

def canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def content_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()

def file_record(path: Path, root: Path) -> dict:
    resolved_root = root.resolve()
    # Validate the target, but retain the logical KITTI frame path for internal symlinks.
    path.resolve().relative_to(resolved_root)
    return {"path": str(path.absolute().relative_to(resolved_root)),
            "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
