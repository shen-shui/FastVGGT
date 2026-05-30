#!/usr/bin/env python3
"""Combine Depth Pro depth maps with FastVGGT camera poses and export density maps."""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.vggt_npz_to_roomformer_density import write_density_outputs
from tools.video_to_roomformer_density import point_colors, write_ply
from vggt.utils.geometry import unproject_depth_map_to_point_map


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Use Depth Pro per-frame depth with FastVGGT extrinsic/intrinsic "
            "to recompute world points and export a RoomFormer density map."
        )
    )
    parser.add_argument("--predictions", type=Path, required=True, help="FastVGGT predictions.npz")
    parser.add_argument("--depthpro-dir", type=Path, required=True, help="Depth Pro output directory")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory")
    parser.add_argument(
        "--depth-key",
        default="depth",
        help="Key inside each Depth Pro .npz file.",
    )
    parser.add_argument(
        "--depth-scale",
        type=float,
        default=1.0,
        help="Multiply Depth Pro depth values by this scale before unprojection.",
    )
    parser.add_argument(
        "--align-depth-scale",
        action="store_true",
        help="Per-frame median-scale Depth Pro depth to FastVGGT depth before unprojection.",
    )
    parser.add_argument(
        "--depth-min",
        type=float,
        default=0.05,
        help="Discard depth values below this threshold after scaling.",
    )
    parser.add_argument(
        "--depth-max",
        type=float,
        default=20.0,
        help="Discard depth values above this threshold after scaling.",
    )
    parser.add_argument(
        "--points-key",
        default="world_points_from_depth",
        choices=["world_points_from_depth", "world_points"],
        help="Fallback FastVGGT point key used only for color/shape validation.",
    )
    parser.add_argument(
        "--conf-key",
        default="depth_conf",
        help="Optional FastVGGT confidence key used to filter recomputed points. Use a missing key such as none to skip.",
    )
    parser.add_argument("--conf-thresh", type=float, default=1.5)
    parser.add_argument(
        "--plane",
        choices=["xy", "xz", "yz", "pca01", "pca02", "pca12", "auto"],
        default="xz",
    )
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--padding", type=float, default=0.05)
    parser.add_argument("--flip-x", action="store_true")
    parser.add_argument("--flip-y", action="store_true")
    parser.add_argument(
        "--max-ply-points",
        type=int,
        default=500000,
        help="Maximum points to write to point_cloud_depthpro.ply. Use 0 to skip PLY.",
    )
    return parser.parse_args()


def sorted_depth_files(depthpro_dir):
    files = sorted(depthpro_dir.rglob("*.npz"))
    if not files:
        raise ValueError(f"No Depth Pro .npz files found in {depthpro_dir}")
    return files


def load_depth_stack(depth_files, depth_key, target_hw, depth_scale, depth_min, depth_max):
    target_h, target_w = target_hw
    depth_maps = []
    for path in depth_files:
        data = np.load(path)
        if depth_key not in data:
            raise KeyError(f"{depth_key!r} not found in {path}")
        depth = np.asarray(data[depth_key], dtype=np.float32) * depth_scale
        if depth.shape != (target_h, target_w):
            depth = cv2.resize(depth, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        valid = np.isfinite(depth) & (depth >= depth_min) & (depth <= depth_max)
        depth = np.where(valid, depth, 0.0).astype(np.float32)
        depth_maps.append(depth[..., None])
    return np.stack(depth_maps, axis=0)


def align_depth_to_vggt(depth_stack, vggt_depth):
    aligned = depth_stack.copy()
    scales = []
    for i in range(aligned.shape[0]):
        src = aligned[i, ..., 0]
        ref = np.asarray(vggt_depth[i]).squeeze().astype(np.float32)
        mask = np.isfinite(src) & np.isfinite(ref) & (src > 1e-6) & (ref > 1e-6)
        if mask.sum() < 100:
            scales.append(1.0)
            continue
        scale = float(np.median(ref[mask]) / max(np.median(src[mask]), 1e-6))
        aligned[i, ..., 0] *= scale
        scales.append(scale)
    return aligned, scales


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    predictions = dict(np.load(args.predictions, allow_pickle=True))
    for key in ("extrinsic", "intrinsic"):
        if key not in predictions:
            raise KeyError(f"{key!r} not found in {args.predictions}")
    if args.points_key not in predictions:
        raise KeyError(f"{args.points_key!r} not found in {args.predictions}")

    reference_points = np.asarray(predictions[args.points_key], dtype=np.float32)
    num_frames, height, width, _ = reference_points.shape
    depth_files = sorted_depth_files(args.depthpro_dir)
    if len(depth_files) != num_frames:
        raise ValueError(
            f"Depth Pro frame count mismatch: found {len(depth_files)} .npz files, "
            f"but predictions contain {num_frames} frames."
        )

    depth_stack = load_depth_stack(
        depth_files,
        depth_key=args.depth_key,
        target_hw=(height, width),
        depth_scale=args.depth_scale,
        depth_min=args.depth_min,
        depth_max=args.depth_max,
    )
    scale_report = None
    if args.align_depth_scale:
        if "depth" not in predictions:
            raise KeyError("--align-depth-scale requires FastVGGT predictions['depth']")
        depth_stack, scale_report = align_depth_to_vggt(depth_stack, predictions["depth"])

    world_points = unproject_depth_map_to_point_map(
        depth_stack,
        np.asarray(predictions["extrinsic"], dtype=np.float32),
        np.asarray(predictions["intrinsic"], dtype=np.float32),
    ).astype(np.float32)

    points_flat = world_points.reshape(-1, 3)
    keep = np.isfinite(points_flat).all(axis=1)
    if args.conf_key in predictions:
        conf = np.asarray(predictions[args.conf_key], dtype=np.float32).reshape(-1)
        if conf.shape[0] == points_flat.shape[0]:
            keep &= np.isfinite(conf) & (conf >= args.conf_thresh)

    points = points_flat[keep]
    if points.shape[0] == 0:
        raise ValueError("No valid points remain after filtering.")

    np.savez_compressed(
        args.output_dir / "predictions_depthpro.npz",
        depth=depth_stack,
        world_points_from_depth=world_points,
        extrinsic=predictions["extrinsic"],
        intrinsic=predictions["intrinsic"],
    )

    selected_plane = write_density_outputs(
        args.output_dir / "density_depthpro.png",
        points,
        plane=args.plane,
        size=args.size,
        padding=args.padding,
        flip_x=args.flip_x,
        flip_y=args.flip_y,
    )
    print(
        f"Wrote {args.output_dir / 'density_depthpro.png'} "
        f"from {points.shape[0]} points (plane={selected_plane})."
    )

    if args.max_ply_points > 0 and "images" in predictions:
        colors = point_colors(predictions, keep)
        write_ply(args.output_dir / "point_cloud_depthpro.ply", points_flat[keep], colors, args.max_ply_points)
        print(f"Wrote {args.output_dir / 'point_cloud_depthpro.ply'}")

    if scale_report is not None:
        print(
            "Depth scale alignment:",
            f"median={np.median(scale_report):.4f}",
            f"min={np.min(scale_report):.4f}",
            f"max={np.max(scale_report):.4f}",
        )


if __name__ == "__main__":
    main()
