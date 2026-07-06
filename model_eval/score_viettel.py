#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Score predicted novel views against ground truth using the VAR 2026 formula.

    Score = 0.4 * (1 - LPIPS) + 0.3 * SSIM + 0.3 * PSNR_norm
    PSNR_norm = clamp(PSNR / PSNR_max, 0, 1)

The leaderboard averages this Score over scenes. This script reuses ArtiFixer's
own metric implementations (``model_eval.metrics_utils.compute_rgb_metrics``:
GenFusion PSNR on [0, 1], window-11 SSIM, LPIPS) so the numbers match what the
rest of the eval stack reports. It is an *estimate* of the leaderboard score:
the contest's exact PSNR_max and LPIPS backbone are set by the organizers and
are exposed here as flags (--psnr_max, --lpips_net).

Two ways to point at predictions:

* ``--mapping image_name_mapping.json`` (from data_processing.test_poses_to_trajectory):
  prediction frames are ``<index:05d>.png`` in --pred_dir and are matched to GT
  file ``<gt_dir>/<image_name>`` by row index. Use this straight off a raw
  ``run_inference`` frame directory.
* no mapping: predictions are matched to GT by filename stem (submission layout,
  where frames were already renamed to the contest image_name).

Single scene:
    python -m model_eval.score_viettel \
        --pred_dir /path/to/artifixer-out/.../hcm0031/frames/batch_0000/pred \
        --gt_dir   /path/to/public_set/hcm0031/test/images \
        --mapping  /path/to/hcm0031/image_name_mapping.json \
        --psnr_max 30

Every public scene at once (predictions already renamed per scene subdir):
    python -m model_eval.score_viettel \
        --pred_root /path/to/predictions \
        --public_root /path/to/public_set \
        --psnr_max 30
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from model_eval.metrics_utils import MetricsAggregator, compute_rgb_metrics, load_image_as_tensor

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".JPG", ".JPEG", ".PNG")
METRIC_NAMES = ("lpips", "ssim", "psnr", "psnr_norm", "score")


def score_from_metrics(metrics: dict[str, float], psnr_max: float) -> tuple[float, float]:
    """Return (psnr_norm, score) for one frame's raw lpips/ssim/psnr metrics."""
    psnr_norm = min(max(metrics["psnr"] / psnr_max, 0.0), 1.0)
    score = 0.4 * (1.0 - metrics["lpips"]) + 0.3 * metrics["ssim"] + 0.3 * psnr_norm
    return psnr_norm, score


def match_by_mapping(pred_dir: Path, gt_dir: Path, mapping_path: Path) -> list[tuple[Path, Path]]:
    mapping = json.loads(mapping_path.read_text())
    pairs = []
    for entry in mapping:
        pred = pred_dir / f"{entry['index']:05d}.png"
        gt = gt_dir / entry["image_name"]
        if not pred.is_file():
            raise FileNotFoundError(f"Missing prediction for index {entry['index']}: {pred}")
        if not gt.is_file():
            raise FileNotFoundError(f"Missing ground-truth image: {gt}")
        pairs.append((pred, gt))
    return pairs


def _index_by_stem(directory: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix in IMAGE_SUFFIXES:
            index[path.stem] = path
    return index


def match_by_stem(pred_dir: Path, gt_dir: Path) -> list[tuple[Path, Path]]:
    preds = _index_by_stem(pred_dir)
    gts = _index_by_stem(gt_dir)
    common = sorted(set(preds) & set(gts))
    missing = sorted(set(gts) - set(preds))
    if missing:
        print(f"  WARNING: {len(missing)} GT frame(s) have no matching prediction, e.g. {missing[:5]}")
    if not common:
        raise ValueError(f"No prediction/GT filename stems match between {pred_dir} and {gt_dir}")
    return [(preds[stem], gts[stem]) for stem in common]


def score_scene(
    scene_id: str,
    pairs: list[tuple[Path, Path]],
    aggregator: MetricsAggregator,
    psnr_max: float,
    lpips_net: str,
    device: torch.device,
) -> None:
    for pred_path, gt_path in pairs:
        pred = load_image_as_tensor(pred_path, device)
        gt = load_image_as_tensor(gt_path, device)
        if pred.shape[-2:] != gt.shape[-2:]:
            print(
                f"  WARNING: {pred_path.name} is {tuple(pred.shape[-2:])} but GT is "
                f"{tuple(gt.shape[-2:])}; resizing prediction to GT size for scoring"
            )
            pred = F.interpolate(pred, size=gt.shape[-2:], mode="bilinear", align_corners=False)
        metrics = compute_rgb_metrics(pred, gt, lpips_net_type=lpips_net)
        psnr_norm, score = score_from_metrics(metrics, psnr_max)
        aggregator.add(
            scene_id,
            lpips=metrics["lpips"],
            ssim=metrics["ssim"],
            psnr=metrics["psnr"],
            psnr_norm=psnr_norm,
            score=score,
        )


def discover_scenes(public_root: Path, requested: list[str] | None) -> list[str]:
    if requested:
        return requested
    scenes = [
        p.name
        for p in sorted(public_root.iterdir())
        if p.is_dir() and (p / "test" / "images").is_dir()
    ]
    if not scenes:
        raise ValueError(f"No scenes with test/images found under {public_root}")
    return scenes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pred_dir", type=Path, help="Prediction directory for a single scene.")
    parser.add_argument("--gt_dir", type=Path, help="Ground-truth test/images directory for a single scene.")
    parser.add_argument("--mapping", type=Path, help="image_name_mapping.json to pair <index>.png preds with GT.")
    parser.add_argument("--scene_id", type=str, default="scene", help="Label for the single-scene report.")

    parser.add_argument("--pred_root", type=Path, help="Batch mode: root with one prediction subdir per scene.")
    parser.add_argument("--public_root", type=Path, help="Batch mode: public_set root (each scene has test/images).")
    parser.add_argument("--scenes", nargs="+", help="Batch mode: restrict to these scene ids.")

    parser.add_argument(
        "--psnr_max",
        type=float,
        default=30.0,
        help="PSNR normalization ceiling for PSNR_norm = clamp(PSNR/PSNR_max, 0, 1). "
        "The contest's exact value is set by the organizers; 30 is a placeholder.",
    )
    parser.add_argument(
        "--lpips_net",
        default="alex",
        choices=["alex", "vgg", "squeeze"],
        help="LPIPS backbone. The contest backbone is unknown; 'alex' is the LPIPS default. "
        "ArtiFixer's own eval uses 'vgg'.",
    )
    parser.add_argument("--results_yaml", type=Path, help="Optional path to write a YAML results summary.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    aggregator = MetricsAggregator(list(METRIC_NAMES))

    batch_mode = args.pred_root is not None or args.public_root is not None
    if batch_mode:
        if args.pred_root is None or args.public_root is None:
            parser.error("batch mode requires both --pred_root and --public_root")
        scenes = discover_scenes(args.public_root, args.scenes)
        for scene_id in scenes:
            pred_dir = args.pred_root / scene_id
            gt_dir = args.public_root / scene_id / "test" / "images"
            if not pred_dir.is_dir():
                print(f"{scene_id}: no prediction dir at {pred_dir}; skipping")
                continue
            pairs = match_by_stem(pred_dir, gt_dir)
            score_scene(scene_id, pairs, aggregator, args.psnr_max, args.lpips_net, device)
            aggregator.print_scene_summary(scene_id, prefix=f"{scene_id}: ")
    else:
        if args.pred_dir is None or args.gt_dir is None:
            parser.error("single-scene mode requires --pred_dir and --gt_dir")
        if args.mapping is not None:
            pairs = match_by_mapping(args.pred_dir, args.gt_dir, args.mapping)
        else:
            pairs = match_by_stem(args.pred_dir, args.gt_dir)
        score_scene(args.scene_id, pairs, aggregator, args.psnr_max, args.lpips_net, device)
        aggregator.print_scene_summary(args.scene_id, prefix=f"{args.scene_id}: ")

    print(f"\nLPIPS backbone: {args.lpips_net} | PSNR_max: {args.psnr_max}")
    print("Overall (mean over all frames):")
    aggregator.print_overall_summary(prefix="  ")

    scene_scores = [
        sum(vals) / len(vals)
        for scene in aggregator.scene_metrics.values()
        if (vals := scene.get("score"))
    ]
    if scene_scores:
        print(f"\nLeaderboard-style Score (mean over {len(scene_scores)} scene(s)): "
              f"{sum(scene_scores) / len(scene_scores):.4f}")

    if args.results_yaml is not None:
        aggregator.save_to_yaml(args.results_yaml)


if __name__ == "__main__":
    main()
