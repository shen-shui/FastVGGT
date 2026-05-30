#!/usr/bin/env python3
"""Convert FastVGGT point outputs to a RoomFormer-style density PNG."""

import argparse
import json
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
        choices=["xy", "xz", "yz", "pca01", "pca02", "pca12", "auto"],
        default="xz",
        help="Top-down plane. Use auto to score axis-aligned and PCA candidates.",
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


def project_points(points, plane):
    if plane in ("xy", "xz", "yz"):
        axes = [AXIS_INDEX[c] for c in plane]
        return points[:, axes].astype(np.float32)

    centered = points.astype(np.float32) - points.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    pca_coords = centered @ vh.T
    pca_axes = {"pca01": (0, 1), "pca02": (0, 2), "pca12": (1, 2)}[plane]
    return pca_coords[:, pca_axes].astype(np.float32)


def coords_to_density(coords, size, padding, flip_x=False, flip_y=False):
    coords = coords.astype(np.float32)

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


def points_to_density(points, plane, size, padding, flip_x=False, flip_y=False):
    coords = project_points(points, plane)
    return coords_to_density(coords, size, padding, flip_x=flip_x, flip_y=flip_y)


def score_density(density):
    mask = density > max(0.02, float(density.max()) * 0.05)
    occupied = int(mask.sum())
    total = int(mask.size)
    occupancy = occupied / max(total, 1)
    if occupied == 0:
        return {
            "score": -1.0,
            "occupancy": 0.0,
            "aspect": 0.0,
            "components": 0,
            "edge_density": 0.0,
            "saturation": 0.0,
        }

    ys, xs = np.nonzero(mask)
    width = int(xs.max() - xs.min() + 1)
    height = int(ys.max() - ys.min() + 1)
    aspect = min(width, height) / max(width, height, 1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    component_areas = stats[1:, cv2.CC_STAT_AREA] if num_labels > 1 else np.array([], dtype=np.int32)
    large_components = int((component_areas > max(20, occupied * 0.01)).sum())
    largest_ratio = float(component_areas.max() / occupied) if component_areas.size else 0.0

    density_u8 = np.clip(density * 255, 0, 255).astype(np.uint8)
    edges = cv2.Canny(density_u8, 20, 80)
    edge_density = float((edges > 0).sum() / max(occupied, 1))
    saturation = float((density >= 0.95).sum() / max(occupied, 1))

    occupancy_score = np.exp(-((occupancy - 0.12) / 0.12) ** 2)
    aspect_score = aspect
    component_score = largest_ratio / max(1.0, large_components)
    edge_score = min(edge_density / 0.35, 1.0)
    saturation_penalty = max(0.0, 1.0 - saturation * 2.0)

    score = (
        0.30 * occupancy_score
        + 0.25 * aspect_score
        + 0.25 * component_score
        + 0.15 * edge_score
        + 0.05 * saturation_penalty
    )
    return {
        "score": float(score),
        "occupancy": float(occupancy),
        "aspect": float(aspect),
        "components": large_components,
        "largest_component_ratio": float(largest_ratio),
        "edge_density": edge_density,
        "saturation": saturation,
    }


def auto_density(points, size, padding, flip_x=False, flip_y=False):
    candidates = {}
    for plane in ("xy", "xz", "yz", "pca01", "pca02", "pca12"):
        density = points_to_density(
            points,
            plane=plane,
            size=size,
            padding=padding,
            flip_x=flip_x,
            flip_y=flip_y,
        )
        metrics = score_density(density)
        candidates[plane] = {"density": density, "metrics": metrics}

    best_plane = max(candidates, key=lambda name: candidates[name]["metrics"]["score"])
    return best_plane, candidates


def write_density_outputs(output_png, points, plane, size, padding, flip_x=False, flip_y=False):
    output_png.parent.mkdir(parents=True, exist_ok=True)

    if plane == "auto":
        best_plane, candidates = auto_density(
            points, size=size, padding=padding, flip_x=flip_x, flip_y=flip_y
        )
        candidates_dir = output_png.parent / "density_candidates"
        candidates_dir.mkdir(parents=True, exist_ok=True)
        report = {"selected": best_plane, "candidates": {}}
        for name, item in candidates.items():
            candidate_path = candidates_dir / f"{name}.png"
            cv2.imwrite(str(candidate_path), (item["density"] * 255).astype(np.uint8))
            report["candidates"][name] = item["metrics"]
        with (output_png.parent / "quality_report.json").open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        density = candidates[best_plane]["density"]
        selected_plane = best_plane
    else:
        density = points_to_density(
            points,
            plane=plane,
            size=size,
            padding=padding,
            flip_x=flip_x,
            flip_y=flip_y,
        )
        selected_plane = plane

    ok = cv2.imwrite(str(output_png), (density * 255).astype(np.uint8))
    if not ok:
        raise OSError(f"Failed to write {output_png}")
    return selected_plane


def main():
    args = parse_args()
    if args.input_path.suffix.lower() == ".ply":
        points = load_ply_points(args.input_path)
    else:
        data = np.load(args.input_path)
        points = load_points(data, args.points_key, args.conf_key, args.conf_thresh)
    selected_plane = write_density_outputs(
        args.output_png,
        points,
        plane=args.plane,
        size=args.size,
        padding=args.padding,
        flip_x=args.flip_x,
        flip_y=args.flip_y,
    )

    print(
        f"Wrote {args.output_png} from {points.shape[0]} points "
        f"({args.size}x{args.size}, plane={selected_plane})."
    )


if __name__ == "__main__":
    main()
