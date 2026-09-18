"""Bounded-memory reader for ScanNet SensorData version 4 files."""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterator, Sequence

import numpy as np


class SensFormatError(ValueError):
    """Raised when a .sens file is unsupported, malformed, or truncated."""


@dataclass(frozen=True)
class SensHeader:
    version: int
    sensor_name: str
    intrinsic_color: np.ndarray
    extrinsic_color: np.ndarray
    intrinsic_depth: np.ndarray
    extrinsic_depth: np.ndarray
    color_compression: int
    depth_compression: int
    color_width: int
    color_height: int
    depth_width: int
    depth_height: int
    depth_shift: float
    frame_count: int

    @property
    def color_size(self) -> tuple[int, int]:
        return (self.color_width, self.color_height)

    @property
    def color_extension(self) -> str:
        try:
            return {1: ".png", 2: ".jpg"}[self.color_compression]
        except KeyError as error:
            raise SensFormatError(
                f"unsupported color compression type {self.color_compression}; expected PNG (1) or JPEG (2)"
            ) from error


@dataclass(frozen=True)
class SensFrame:
    frame_id: int
    camera_to_world: np.ndarray
    timestamp_color: int
    timestamp_depth: int
    color_offset: int
    color_size: int
    depth_offset: int
    depth_size: int


def _read_exact(stream: BinaryIO, size: int, label: str) -> bytes:
    data = stream.read(size)
    if len(data) != size:
        raise SensFormatError(
            f"truncated .sens while reading {label}: expected {size} bytes, got {len(data)}"
        )
    return data


def _unpack(stream: BinaryIO, fmt: str, label: str) -> tuple[object, ...]:
    size = struct.calcsize(fmt)
    return struct.unpack(fmt, _read_exact(stream, size, label))


def _matrix(stream: BinaryIO, label: str) -> np.ndarray:
    return np.frombuffer(_read_exact(stream, 64, label), dtype="<f4").reshape(4, 4).copy()


class SensReader:
    """Parse a .sens header and frame index without loading image payloads."""

    _MAX_SENSOR_NAME_BYTES = 1024 * 1024

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)
        self._file_size = self.path.stat().st_size
        with self.path.open("rb") as stream:
            self.header = self._read_header(stream)
            self.frames, self.imu_frame_count = self._read_frame_index(stream, self.header.frame_count)

    def _read_header(self, stream: BinaryIO) -> SensHeader:
        (version,) = _unpack(stream, "<I", "version")
        if version != 4:
            raise SensFormatError(f"unsupported .sens version {version}; expected version 4")
        (name_size,) = _unpack(stream, "<Q", "sensor name length")
        if name_size > self._MAX_SENSOR_NAME_BYTES:
            raise SensFormatError(f"sensor name length is unreasonable: {name_size}")
        name_bytes = _read_exact(stream, int(name_size), "sensor name")
        try:
            sensor_name = name_bytes.decode("utf-8")
        except UnicodeDecodeError as error:
            raise SensFormatError("sensor name is not valid UTF-8") from error

        intrinsic_color = _matrix(stream, "intrinsic_color")
        extrinsic_color = _matrix(stream, "extrinsic_color")
        intrinsic_depth = _matrix(stream, "intrinsic_depth")
        extrinsic_depth = _matrix(stream, "extrinsic_depth")
        values = _unpack(stream, "<iiIIIIfQ", "sensor metadata")
        (
            color_compression,
            depth_compression,
            color_width,
            color_height,
            depth_width,
            depth_height,
            depth_shift,
            frame_count,
        ) = values
        if color_width <= 0 or color_height <= 0 or depth_width <= 0 or depth_height <= 0:
            raise SensFormatError("sensor image dimensions must be positive")
        if not np.isfinite(depth_shift) or depth_shift <= 0:
            raise SensFormatError("depth_shift must be finite and positive")
        minimum_frame_bytes = 96
        remaining = self._file_size - stream.tell()
        if frame_count > remaining // minimum_frame_bytes:
            raise SensFormatError(
                f"truncated .sens frame table: {frame_count} frames cannot fit in {remaining} bytes"
            )
        header = SensHeader(
            version=int(version),
            sensor_name=sensor_name,
            intrinsic_color=intrinsic_color,
            extrinsic_color=extrinsic_color,
            intrinsic_depth=intrinsic_depth,
            extrinsic_depth=extrinsic_depth,
            color_compression=int(color_compression),
            depth_compression=int(depth_compression),
            color_width=int(color_width),
            color_height=int(color_height),
            depth_width=int(depth_width),
            depth_height=int(depth_height),
            depth_shift=float(depth_shift),
            frame_count=int(frame_count),
        )
        _ = header.color_extension
        return header

    def _read_frame_index(
        self, stream: BinaryIO, frame_count: int
    ) -> tuple[tuple[SensFrame, ...], int]:
        frames: list[SensFrame] = []
        for frame_id in range(frame_count):
            pose = _matrix(stream, f"frame {frame_id} camera_to_world")
            timestamp_color, timestamp_depth, color_size, depth_size = _unpack(
                stream, "<QQQQ", f"frame {frame_id} metadata"
            )
            color_offset = stream.tell()
            depth_offset = color_offset + int(color_size)
            end_offset = depth_offset + int(depth_size)
            if end_offset > self._file_size:
                raise SensFormatError(
                    f"truncated .sens payload for frame {frame_id}: payload ends at {end_offset}, file size is {self._file_size}"
                )
            stream.seek(end_offset)
            frames.append(
                SensFrame(
                    frame_id=frame_id,
                    camera_to_world=pose,
                    timestamp_color=int(timestamp_color),
                    timestamp_depth=int(timestamp_depth),
                    color_offset=color_offset,
                    color_size=int(color_size),
                    depth_offset=depth_offset,
                    depth_size=int(depth_size),
                )
            )
        remaining = self._file_size - stream.tell()
        imu_frame_count = 0
        if remaining:
            if remaining < 8:
                raise SensFormatError(f"truncated .sens IMU trailer: only {remaining} bytes remain")
            (imu_frame_count,) = _unpack(stream, "<Q", "IMU frame count")
            expected_imu_bytes = int(imu_frame_count) * 128
            actual_imu_bytes = self._file_size - stream.tell()
            if actual_imu_bytes != expected_imu_bytes:
                raise SensFormatError(
                    f"malformed .sens IMU trailer: count {imu_frame_count} requires "
                    f"{expected_imu_bytes} bytes, found {actual_imu_bytes}"
                )
            stream.seek(expected_imu_bytes, os.SEEK_CUR)
        return tuple(frames), int(imu_frame_count)

    def _validate_frame(self, frame: SensFrame) -> None:
        if frame.frame_id < 0 or frame.frame_id >= len(self.frames) or self.frames[frame.frame_id] is not frame:
            raise ValueError("frame does not belong to this .sens reader")

    def iter_color_payloads(self, frames: Sequence[SensFrame]) -> Iterator[bytes]:
        """Yield encoded color payloads using one file handle."""

        with self.path.open("rb") as stream:
            for frame in frames:
                self._validate_frame(frame)
                stream.seek(frame.color_offset)
                yield _read_exact(stream, frame.color_size, f"frame {frame.frame_id} color payload")

    def read_color_payload(self, frame: SensFrame) -> bytes:
        return next(self.iter_color_payloads((frame,)))


def sample_frame_ids(frame_ids: Sequence[int], max_frames: int | None = None) -> tuple[int, ...]:
    """Apply FastVGGT's deterministic first-frame-plus-stride policy."""

    ordered = tuple(frame_ids)
    if max_frames is not None and (
        isinstance(max_frames, bool) or not isinstance(max_frames, int) or max_frames <= 0
    ):
        raise ValueError("max_frames must be None or a positive integer")
    if max_frames is None or len(ordered) <= max_frames:
        return ordered
    if max_frames == 1:
        return ordered[:1]
    remaining = ordered[1:]
    step = max(1, len(remaining) // (max_frames - 1))
    return ordered[:1] + remaining[::step][: max_frames - 1]


def select_frame_ids(frames: Sequence[SensFrame], max_frames: int | None = None) -> tuple[int, ...]:
    """Filter non-finite poses, then apply FastVGGT's selection policy."""

    if max_frames is not None and (
        isinstance(max_frames, bool) or not isinstance(max_frames, int) or max_frames <= 0
    ):
        raise ValueError("max_frames must be None or a positive integer")
    valid = [frame.frame_id for frame in frames if np.isfinite(frame.camera_to_world).all()]
    return sample_frame_ids(valid, max_frames=max_frames)


__all__ = ["SensFormatError", "SensFrame", "SensHeader", "SensReader", "sample_frame_ids", "select_frame_ids"]
