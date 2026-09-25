#!/usr/bin/env python3
"""Move known AutoDL experiment roots below one canonical results directory."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Iterator


EXPERIMENT_ROOTS = (
    "camera_context",
    "camera_iteration",
    "camera_head_amplification",
    "camera_hidden_state_attribution",
    "camera_hidden_causal_preference",
    "camera_hidden_replacement",
    "camera_hidden_adaptive_alpha",
    "local_global_consistency",
    "camera_refiner_data_construction",
    "vggt_hallucination",
    "camera_refiner_training",
)


class MigrationConflict(ValueError):
    """Raised when migration would overwrite or reinterpret existing data."""


@dataclass(frozen=True)
class RootMigration:
    name: str
    legacy: Path
    canonical: Path
    action: str


@dataclass(frozen=True)
class MigrationPlan:
    autodl_tmp: Path
    results_root: Path
    roots: tuple[RootMigration, ...]


def _lexists(path: Path) -> bool:
    return os.path.lexists(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path_kind(path: Path) -> str:
    if path.is_symlink():
        return "symlink"
    if path.is_dir():
        return "directory"
    if path.is_file():
        return "file"
    return "unsupported"


def _relative_label(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _validate_merge(source: Path, destination: Path, canonical_root: Path) -> None:
    for source_child in sorted(source.iterdir(), key=lambda item: item.name):
        destination_child = destination / source_child.name
        source_kind = _path_kind(source_child)
        label = _relative_label(destination_child, canonical_root)
        if source_kind == "unsupported":
            raise MigrationConflict(f"unsupported source entry: {label}")
        if not _lexists(destination_child):
            continue
        destination_kind = _path_kind(destination_child)
        if source_kind != destination_kind:
            raise MigrationConflict(
                f"filesystem type conflict at {label}: "
                f"{source_kind} != {destination_kind}"
            )
        if source_kind == "directory":
            _validate_merge(source_child, destination_child, canonical_root)
        elif source_kind == "file":
            if source_child.stat().st_size != destination_child.stat().st_size or (
                _sha256(source_child) != _sha256(destination_child)
            ):
                raise MigrationConflict(f"file content conflict at {label}")
        elif os.readlink(source_child) != os.readlink(destination_child):
            raise MigrationConflict(f"symlink target conflict at {label}")


def _resolved_link(path: Path) -> Path:
    target = Path(os.readlink(path))
    if not target.is_absolute():
        target = path.parent / target
    return target.resolve(strict=False)


def plan_migration(autodl_tmp: Path, results_root: Path) -> MigrationPlan:
    """Validate every approved root and return a mutation-free migration plan."""
    root = Path(autodl_tmp).expanduser().resolve(strict=False)
    canonical_root = Path(results_root).expanduser().resolve(strict=False)
    if root == canonical_root:
        raise MigrationConflict("results root must differ from AUTODL_TMP")
    for name in EXPERIMENT_ROOTS:
        legacy_candidate = (root / name).resolve(strict=False)
        if canonical_root == legacy_candidate or legacy_candidate in canonical_root.parents:
            raise MigrationConflict(
                f"results root cannot be nested under legacy root {name}"
            )

    roots = []
    for name in EXPERIMENT_ROOTS:
        legacy = root / name
        canonical = canonical_root / name
        if legacy.is_symlink():
            if _resolved_link(legacy) != canonical.resolve(strict=False):
                raise MigrationConflict(
                    f"legacy symlink for {name} does not target {canonical}"
                )
            if not canonical.is_dir() or canonical.is_symlink():
                raise MigrationConflict(
                    f"legacy symlink for {name} targets a missing or invalid directory"
                )
            action = "linked"
        else:
            if _lexists(legacy) and not legacy.is_dir():
                raise MigrationConflict(f"legacy root is not a directory: {legacy}")
            if canonical.is_symlink() or (
                _lexists(canonical) and not canonical.is_dir()
            ):
                raise MigrationConflict(
                    f"canonical root is not a real directory: {canonical}"
                )
            legacy_exists = legacy.is_dir()
            canonical_exists = canonical.is_dir()
            if legacy_exists and canonical_exists:
                _validate_merge(legacy, canonical, canonical)
                action = "merge"
            elif legacy_exists:
                action = "move"
            elif canonical_exists:
                action = "link"
            else:
                action = "absent"
        roots.append(
            RootMigration(
                name=name,
                legacy=legacy,
                canonical=canonical,
                action=action,
            )
        )
    return MigrationPlan(root, canonical_root, tuple(roots))


def _merge_directories(source: Path, destination: Path) -> int:
    identical_removed = 0
    for source_child in sorted(source.iterdir(), key=lambda item: item.name):
        destination_child = destination / source_child.name
        if not _lexists(destination_child):
            source_child.rename(destination_child)
        elif source_child.is_dir() and not source_child.is_symlink():
            identical_removed += _merge_directories(source_child, destination_child)
        else:
            source_child.unlink()
            identical_removed += 1
    source.rmdir()
    return identical_removed


def _create_compatibility_link(item: RootMigration) -> None:
    if item.legacy.is_symlink():
        return
    if _lexists(item.legacy):
        raise MigrationConflict(
            f"cannot create compatibility link over existing path: {item.legacy}"
        )
    item.legacy.symlink_to(item.canonical, target_is_directory=True)


def execute_migration(
    plan: MigrationPlan,
    *,
    dry_run: bool = False,
    create_links: bool = True,
) -> dict[str, object]:
    """Execute a fresh validated plan without overwriting existing artifacts."""
    fresh = plan_migration(plan.autodl_tmp, plan.results_root)
    migrated = [
        item.name for item in fresh.roots if item.action in {"move", "merge"}
    ]
    canonical = [
        item.name for item in fresh.roots if item.action in {"link", "linked"}
    ]
    links = [item.name for item in fresh.roots if item.action == "linked"]
    identical_removed = 0
    if not dry_run:
        fresh.results_root.mkdir(parents=True, exist_ok=True)
        for item in fresh.roots:
            if item.action == "move":
                item.canonical.parent.mkdir(parents=True, exist_ok=True)
                item.legacy.rename(item.canonical)
            elif item.action == "merge":
                identical_removed += _merge_directories(
                    item.legacy,
                    item.canonical,
                )
            if create_links and item.action in {"move", "merge", "link"}:
                _create_compatibility_link(item)
                links.append(item.name)
    return {
        "schema_version": 1,
        "dry_run": bool(dry_run),
        "autodl_tmp": str(fresh.autodl_tmp),
        "results_root": str(fresh.results_root),
        "migrated_roots": migrated,
        "canonical_roots": canonical,
        "compatibility_links": links,
        "identical_files_removed": identical_removed,
    }


@contextmanager
def _exclusive_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        if os.name == "posix":
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield
        if os.name == "posix":
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--autodl-tmp",
        type=Path,
        default=Path(os.environ.get("AUTODL_TMP", "/root/autodl-tmp")),
    )
    parser.add_argument("--results-root", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    autodl_tmp = args.autodl_tmp.expanduser().resolve(strict=False)
    results_root = (
        args.results_root
        if args.results_root is not None
        else Path(os.environ.get("RESULTS_ROOT", str(autodl_tmp / "results")))
    ).expanduser().resolve(strict=False)
    if args.dry_run:
        report = execute_migration(
            plan_migration(autodl_tmp, results_root),
            dry_run=True,
        )
    else:
        with _exclusive_lock(results_root / ".migration.lock"):
            report = execute_migration(
                plan_migration(autodl_tmp, results_root),
            )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
