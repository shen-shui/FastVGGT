#!/usr/bin/env python3
"""Convert FastVGGT point outputs to a RoomFormer-style density PNG."""

import argparse
from pathlib import Path

import cv2
import numpy as np


AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Project FastVGGT 3D points to a 256x256 RoomFormer density map."
    )
    parser.add_argument("input_path", type=Path, help="FastVGGT predictions.npz or PLY")
    parser.add_argument("output_png", type=Path, help="Output grayscale density PNG")
    parser.add_argument(
        "--points-key",
        default="world_points_from_depth",
        help="NPZ key containing points shaped (S,H,W,3) or (N,3).",
    )
    parser.add_argument(
        "--conf-key",
        default="depth_conf",
        help="Optional NPZ key containing confidence shaped (S,H,W).",
    )
    parser.add_argument(
        "--conf-thresh",
        type=float,
        default=3.0,
        help="Keep points with confidence >= this value when confidence exists.",
    )
    parser.add_argument(
        "--plane",
        choices=["xy", "xz", "yz"],
        default="xz",
        help="Top-down plane. VGGT/OpenCV-style outputs usually use xz.",
    )
    parser.add_argument("--size", type=int, default=256, help="Output image size.")
    parser.add_argument(
        "--padding",
        type=float,
        default=0.05,
        help="Square-bound padding ratio, matching RoomFormer SceneCAD preprocessing.",
    )
    parser.add_argument(
        "--flip-x",
        action="store_true",
        help="Flip horizontal image axis after projection.",
    )
    parser.add_argument(
        "--flip-y",
        action="store_true",
        help="Flip vertical image axis after projection.",
    )
    return parser.parse_args()


def load_points(data, points_key, conf_key, conf_thresh):
    if points_key not in data:
        keys = ", ".join(data.files)
        raise KeyError(f"{points_key!r} not found in {keys}")

    points = np.asarray(data[points_key], dtype=np.float32)
    points = points.reshape(-1, 3)

    mask = np.isfinite(points).all(axis=1)
    if conf_key in data:
        conf = np.asarray(data[conf_key], dtype=np.float32).reshape(-1)
        if conf.shape[0] == points.shape[0]:
            mask &= np.isfinite(conf) & (conf >= conf_thresh)

    points = points[mask]
    if points.shape[0] == 0:
        raise ValueError("No valid points remain after filtering.")
    return points


def load_ply_points(path):
    try:
        from plyfile import PlyData
    except ImportError as exc:
        raise ImportError("Reading PLY requires `pip install plyfile`.") from exc

    with path.open("rb") as handle:
        ply = PlyData.read(handle)
    vertex = ply["vertex"].data
    points = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float32)
    mask = np.isfinite(points).all(axis=1)
    points = points[mask]
    if points.shape[0] == 0:
        raise ValueError("No valid points found in the PLY file.")
    return points


def points_to_density(points, plane, size, padding, flip_x=False, flip_y=False):
    axes = [AXIS_INDEX[c] for c in plane]
    coords = points[:, axes].astype(np.float32)

    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    center = (mins + maxs) * 0.5
    max_range = float((maxs - mins).max())
    if max_range <= 0:
        raise ValueError("Point cloud has zero extent on the selected plane.")

    max_range *= 1.0 + 2.0 * padding
    mins = center - max_range * 0.5

    normalized = (coords - mins[None, :]) / max_range
    pixels = np.round(normalized * size)
    pixels = np.clip(pixels, 0, size - 1).astype(np.int32)

    if flip_x:
        pixels[:, 0] = size - 1 - pixels[:, 0]
    if flip_y:
        pixels[:, 1] = size - 1 - pixels[:, 1]

    density = np.zeros((size, size), dtype=np.float32)
    unique_pixels, counts = np.unique(pixels, return_counts=True, axis=0)
    density[unique_pixels[:, 1], unique_pixels[:, 0]] = counts.astype(np.float32)
    density /= max(float(density.max()), 1.0)
    return density


def main():
    args = parse_args()
    if args.input_path.suffix.lower() == ".ply":
        points = load_ply_points(args.input_path)
    else:
        data = np.load(args.input_path)
        points = load_points(data, args.points_key, args.conf_key, args.conf_thresh)
    density = points_to_density(
        points,
        plane=args.plane,
        size=args.size,
        padding=args.padding,
        flip_x=args.flip_x,
        flip_y=args.flip_y,
    )

    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(args.output_png), (density * 255).astype(np.uint8))
    if not ok:
        raise OSError(f"Failed to write {args.output_png}")

    print(
        f"Wrote {args.output_png} from {points.shape[0]} points "
        f"({args.size}x{args.size}, plane={args.plane})."
    )


if __name__ == "__main__":
    main()
