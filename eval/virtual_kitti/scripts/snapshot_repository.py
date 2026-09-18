#!/usr/bin/env python3
"""Deterministic local content/identity snapshot; never follow repository symlinks."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import sys

# Only benign tooling scratch and the documented optional dependency target.
CACHE_NAMES = {".git", ".superpowers", ".pytest_cache", "__pycache__",
               ".mypy_cache", ".ruff_cache"}


def excluded(parts: tuple[str, ...]) -> bool:
    return (any(part in CACHE_NAMES for part in parts)
            or parts[-1].endswith((".pyc", ".pyo"))
            or parts[:2] == (".runtime", "long_deps"))


def identity(info: os.stat_result) -> list[int]:
    return [info.st_dev, info.st_ino, info.st_mode]


def stable_file(info: os.stat_result) -> tuple[int, ...]:
    return (*identity(info), info.st_size, info.st_nlink,
            info.st_mtime_ns, info.st_ctime_ns)


def snapshot(root: Path) -> dict:
    entries = {}

    def walk(directory_fd: int, prefix: tuple[str, ...]) -> None:
        for name in sorted(os.listdir(directory_fd)):
            parts = (*prefix, name)
            if excluded(parts):
                continue
            relative = "/".join(parts)
            info = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            record = {"identity": identity(info)}
            if stat.S_ISLNK(info.st_mode):
                record.update(kind="symlink", target=os.readlink(name, dir_fd=directory_fd))
            elif stat.S_ISDIR(info.st_mode):
                record["kind"] = "directory"
                child_fd = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                                   dir_fd=directory_fd)
                try:
                    if identity(os.fstat(child_fd)) != identity(info):
                        raise RuntimeError(f"Directory changed while snapshotting: {relative}")
                    walk(child_fd, parts)
                finally:
                    os.close(child_fd)
            elif stat.S_ISREG(info.st_mode):
                digest = hashlib.sha256()
                descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                                     dir_fd=directory_fd)
                with os.fdopen(descriptor, "rb") as stream:
                    opened = os.fstat(stream.fileno())
                    if stable_file(opened) != stable_file(info):
                        raise RuntimeError(f"File changed while snapshotting: {relative}")
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
                    if stable_file(os.fstat(stream.fileno())) != stable_file(opened):
                        raise RuntimeError(f"File changed while snapshotting: {relative}")
                record.update(kind="file", size=info.st_size, sha256=digest.hexdigest())
            else:
                # Record special filesystem objects without opening/blocking on them.
                record["kind"] = "special"
            current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            if identity(current) != identity(info):
                raise RuntimeError(f"Entry changed while snapshotting: {relative}")
            entries[relative] = record

    directory_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        walk(directory_fd, ())
    finally:
        os.close(directory_fd)
    return {"schema_version": 1, "entries": entries}


if __name__ == "__main__":
    print(json.dumps(snapshot(Path(sys.argv[1])), sort_keys=True, indent=2, allow_nan=False))
