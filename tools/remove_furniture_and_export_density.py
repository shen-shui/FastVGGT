#!/usr/bin/env python3
"""Remove segmented furniture points from FastVGGT outputs and export density maps."""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.vggt_npz_to_roomformer_density import write_density_outputs
from tools.video_to_roomformer_density import point_colors, write_ply


DEFAULT_REMOVE_CLASSES = (
    "bed",
    "chair",
    "couch",
    "dining table",
    "tv",
    "potted plant",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Use YOLOv8-seg masks to remove furniture pixels from FastVGGT "
            "world points, then export a RoomFormer-style wall density PNG."
        )
    )
    parser.add_argument("--predictions", type=Path, required=True, help="FastVGGT predictions.npz")
    parser.add_argument(
        "--frames-dir",
        type=Path,
        required=True,
        help="Directory containing sampled frames used by the predictions.",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory.")
    parser.add_argument(
        "--seg-model",
        default="yolov8x-seg.pt",
        help="Ultralytics YOLO segmentation model path or model name.",
    )
    parser.add_argument(
        "--remove-classes",
        nargs="+",
        default=list(DEFAULT_REMOVE_CLASSES),
        help="Class names to remove from the 3D point cloud.",
    )
    parser.add_argument(
        "--seg-conf",
        type=float,
        default=0.25,
        help="YOLO segmentation confidence threshold.",
    )
    parser.add_argument(
        "--mask-dilate",
        type=int,
        default=3,
        help="Dilate furniture masks by this many pixels at VGGT output resolution.",
    )
    parser.add_argument(
        "--points-key",
        default="world_points_from_depth",
        choices=["world_points_from_depth", "world_points"],
        help="Point set used for filtering and density export.",
    )
    parser.add_argument(
        "--conf-key",
        default="depth_conf",
        help="Optional confidence key used after furniture removal. Use a missing key such as none to skip.",
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
        help="Maximum filtered points to write to point_cloud_filtered.ply. Use 0 to skip PLY.",
    )
    parser.add_argument(
        "--save-mask-debug",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save per-frame furniture mask overlays for inspection.",
    )
    return parser.parse_args()


def load_frame_paths(frames_dir):
    paths = sorted(
        p
        for p in frames_dir.iterdir()
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
    )
    if not paths:
        raise ValueError(f"No image frames found in {frames_dir}")
    return paths


def class_ids_from_names(model, class_names):
    requested = {name.lower() for name in class_names}
    names = model.names
    matched = {}
    for class_id, name in names.items():
        if str(name).lower() in requested:
            matched[int(class_id)] = str(name)
    missing = sorted(requested - {name.lower() for name in matched.values()})
    return matched, missing


def resize_binary_mask(mask, size_hw):
    height, width = size_hw
    resized = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
    return resized.astype(bool)


def dilate_mask(mask, radius):
    if radius <= 0:
        return mask
    kernel_size = radius * 2 + 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)


def write_mask_debug(path, frame_path, furniture_mask):
    frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
    if frame is None:
        return
    overlay_mask = cv2.resize(
        furniture_mask.astype(np.uint8),
        (frame.shape[1], frame.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)
    overlay = frame.copy()
    overlay[overlay_mask] = (0, 0, 255)
    blended = cv2.addWeighted(frame, 0.65, overlay, 0.35, 0.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), blended)


def furniture_masks_for_frames(frame_paths, output_hw, model_name, remove_classes, seg_conf, dilate, debug_dir):
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "Furniture removal requires ultralytics. Install it with: "
            'uv pip install --python "$(which python)" ultralytics opencv-python'
        ) from exc

    model = YOLO(model_name)
    remove_ids, missing = class_ids_from_names(model, remove_classes)
    if not remove_ids:
        raise ValueError(
            "None of the requested classes exist in this YOLO model. "
            f"Requested: {', '.join(remove_classes)}"
        )
    if missing:
        print(f"Warning: classes not found in model and will be ignored: {', '.join(missing)}")
    print("Removing classes:", ", ".join(f"{name}({idx})" for idx, name in remove_ids.items()))

    masks = []
    detections = []
    for frame_index, frame_path in enumerate(frame_paths):
        result = model.predict(str(frame_path), conf=seg_conf, verbose=False)[0]
        furniture_mask = np.zeros(output_hw, dtype=bool)
        frame_detections = []

        if result.masks is not None and result.boxes is not None:
            cls_ids = result.boxes.cls.detach().cpu().numpy().astype(int)
            mask_data = result.masks.data.detach().cpu().numpy()
            for det_index, class_id in enumerate(cls_ids):
                if class_id not in remove_ids:
                    continue
                mask = resize_binary_mask(mask_data[det_index] > 0.5, output_hw)
                furniture_mask |= mask
                frame_detections.append(remove_ids[class_id])

        furniture_mask = dilate_mask(furniture_mask, dilate)
        masks.append(furniture_mask)
        detections.append(frame_detections)

        if debug_dir is not None:
            write_mask_debug(debug_dir / f"{frame_index:06d}.png", frame_path, furniture_mask)

    return np.stack(masks, axis=0), detections, remove_ids, missing


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(args.predictions, allow_pickle=True)
    predictions = {key: data[key] for key in data.files}
    if args.points_key not in predictions:
        raise KeyError(f"{args.points_key!r} not found in {args.predictions}")

    points_grid = np.asarray(predictions[args.points_key], dtype=np.float32)
    if points_grid.ndim != 4 or points_grid.shape[-1] != 3:
        raise ValueError(f"{args.points_key} must have shape (N,H,W,3), got {points_grid.shape}")

    num_frames, height, width, _ = points_grid.shape
    frame_paths = load_frame_paths(args.frames_dir)
    if len(frame_paths) != num_frames:
        raise ValueError(
            f"Frame count mismatch: {len(frame_paths)} frames in {args.frames_dir}, "
            f"but {args.points_key} has {num_frames} frames."
        )

    debug_dir = args.output_dir / "mask_debug" if args.save_mask_debug else None
    furniture_masks, detections, remove_ids, missing = furniture_masks_for_frames(
        frame_paths=frame_paths,
        output_hw=(height, width),
        model_name=args.seg_model,
        remove_classes=args.remove_classes,
        seg_conf=args.seg_conf,
        dilate=args.mask_dilate,
        debug_dir=debug_dir,
    )

    points_flat = points_grid.reshape(-1, 3)
    valid = np.isfinite(points_flat).all(axis=1)
    furniture_flat = furniture_masks.reshape(-1)
    keep = valid & ~furniture_flat

    if args.conf_key in predictions:
        conf = np.asarray(predictions[args.conf_key], dtype=np.float32).reshape(-1)
        if conf.shape[0] == points_flat.shape[0]:
            keep &= np.isfinite(conf) & (conf >= args.conf_thresh)

    points = points_flat[keep]
    if points.shape[0] == 0:
        raise ValueError("No points remain after furniture and confidence filtering.")

    density_path = args.output_dir / "wall_density.png"
    selected_plane = write_density_outputs(
        density_path,
        points,
        plane=args.plane,
        size=args.size,
        padding=args.padding,
        flip_x=args.flip_x,
        flip_y=args.flip_y,
    )

    report = {
        "predictions": str(args.predictions),
        "frames_dir": str(args.frames_dir),
        "seg_model": args.seg_model,
        "requested_remove_classes": args.remove_classes,
        "matched_remove_classes": remove_ids,
        "missing_remove_classes": missing,
        "num_frames": num_frames,
        "points_total": int(points_flat.shape[0]),
        "points_finite": int(valid.sum()),
        "points_in_furniture_masks": int((valid & furniture_flat).sum()),
        "points_kept": int(points.shape[0]),
        "selected_plane": selected_plane,
        "detections_per_frame": detections,
    }
    with (args.output_dir / "furniture_filter_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(f"Wrote {density_path} from {points.shape[0]} filtered points (plane={selected_plane}).")
    print(f"Wrote {args.output_dir / 'furniture_filter_report.json'}")

    if args.max_ply_points > 0 and "images" in predictions:
        colors = point_colors(predictions, keep)
        write_ply(args.output_dir / "point_cloud_filtered.ply", points_flat[keep], colors, args.max_ply_points)
        print(f"Wrote {args.output_dir / 'point_cloud_filtered.ply'}")


if __name__ == "__main__":
    main()
