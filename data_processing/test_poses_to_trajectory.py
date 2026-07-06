#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert a Viettel VAR contest ``test_poses.csv`` into an ArtiFixer trajectory JSON.

``test_poses.csv`` lists one target camera per row in COLMAP convention
(``qw,qx,qy,qz,tx,ty,tz`` world-to-camera, ``fx,fy,cx,cy,width,height``
intrinsics). This writes:

* a target-only trajectory JSON consumable by
  ``data_processing.prepare_colmap_artifixer_inputs --trajectory_path``,
  which renders frames as ``<index:05d>.png`` in row order, and
* an index -> ``image_name`` mapping so a later submission-packaging step can
  rename those renders to the filenames the contest requires.

Kept free of torch/threedgrut imports so this conversion can run outside the
training/reconstruction environment.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from data_processing.camera_trajectories import (
    opencv_w2c_to_opengl_c2w,
    transforms_json,
    write_json,
)

REQUIRED_CSV_COLUMNS = (
    "image_name",
    "qw",
    "qx",
    "qy",
    "qz",
    "tx",
    "ty",
    "tz",
    "fx",
    "fy",
    "cx",
    "cy",
    "width",
    "height",
)


def qvec_to_so3(qvec: np.ndarray) -> np.ndarray:
    """COLMAP unit-quaternion (qw, qx, qy, qz) to a 3x3 rotation matrix."""
    qw, qx, qy, qz = qvec
    return np.array(
        [
            [1 - 2 * qy**2 - 2 * qz**2, 2 * qx * qy - 2 * qw * qz, 2 * qz * qx + 2 * qw * qy],
            [2 * qx * qy + 2 * qw * qz, 1 - 2 * qx**2 - 2 * qz**2, 2 * qy * qz - 2 * qw * qx],
            [2 * qz * qx - 2 * qw * qy, 2 * qy * qz + 2 * qw * qx, 1 - 2 * qx**2 - 2 * qy**2],
        ]
    )


def read_test_poses_csv(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.expanduser().open(newline="") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames is not None, f"{csv_path} has no header row"
        missing = [col for col in REQUIRED_CSV_COLUMNS if col not in reader.fieldnames]
        assert not missing, f"{csv_path} is missing required columns: {missing}"
        rows = list(reader)
    assert rows, f"{csv_path} contains no pose rows"
    return rows


def row_to_camera_to_world(row: dict[str, str]) -> list[list[float]]:
    qvec = np.array([float(row["qw"]), float(row["qx"]), float(row["qy"]), float(row["qz"])])
    tvec = np.array([float(row["tx"]), float(row["ty"]), float(row["tz"])])
    world_to_camera = np.eye(4, dtype=np.float64)
    world_to_camera[:3, :3] = qvec_to_so3(qvec)
    world_to_camera[:3, 3] = tvec
    return opencv_w2c_to_opengl_c2w(world_to_camera).tolist()


def row_intrinsics(row: dict[str, str]) -> dict[str, float | int]:
    return {
        "fl_x": float(row["fx"]),
        "fl_y": float(row["fy"]),
        "cx": float(row["cx"]),
        "cy": float(row["cy"]),
        "w": int(float(row["width"])),
        "h": int(float(row["height"])),
    }


def build_trajectory_and_mapping(rows: list[dict[str, str]]) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Build the trajectory JSON payload and the index -> image_name mapping.

    Row order is preserved: row ``i`` becomes trajectory frame ``i``, and
    ``prepare_colmap_artifixer_inputs`` renders trajectory frame ``i`` as
    ``<i:05d>.png``, so ``mapping[i]`` tells a submission-packaging step which
    contest filename that render corresponds to.
    """
    top_intrinsics = {"camera_model": "OPENCV", **row_intrinsics(rows[0])}

    frames = []
    mapping = []
    for index, row in enumerate(rows):
        frame_intrinsics = row_intrinsics(row)
        frame: dict[str, object] = {"transform_matrix": row_to_camera_to_world(row)}
        if frame_intrinsics != {k: top_intrinsics[k] for k in frame_intrinsics}:
            frame.update(frame_intrinsics)
        frames.append(frame)
        mapping.append({"index": index, "image_name": row["image_name"], **frame_intrinsics})

    trajectory = transforms_json(top_intrinsics, frames)
    return trajectory, mapping


def convert(csv_path: Path, output_trajectory: Path, output_mapping: Path) -> None:
    rows = read_test_poses_csv(csv_path)
    trajectory, mapping = build_trajectory_and_mapping(rows)
    write_json(output_trajectory, trajectory)
    write_json(output_mapping, mapping)
    print(f"Wrote {len(rows)} target frames to {output_trajectory}")
    print(f"Wrote image_name mapping to {output_mapping}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test_poses_csv", required=True, type=Path)
    parser.add_argument(
        "--output_trajectory",
        required=True,
        type=Path,
        help="Where to write the trajectory JSON for --trajectory_path.",
    )
    parser.add_argument(
        "--output_mapping",
        required=True,
        type=Path,
        help="Where to write the index -> image_name mapping JSON for submission packaging.",
    )
    args = parser.parse_args()
    convert(args.test_poses_csv, args.output_trajectory, args.output_mapping)


if __name__ == "__main__":
    main()
