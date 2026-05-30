#!/usr/bin/env python3
"""Filter FastVGGT points with ADE20K semantic segmentation and export density maps."""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.vggt_npz_to_roomformer_density import write_density_outputs
from tools.video_to_roomformer_density import point_colors, write_ply


DEFAULT_KEEP_CLASSES = (
    "wall",
    "floor",
    "ceiling",
    "door",
    "window",
    "windowpane",
)

DEFAULT_REMOVE_CLASSES = (
    "bed",
    "cabinet",
    "wardrobe",
    "shelf",
    "bookcase",
    "desk",
    "table",
    "chair",
    "sofa",
    "curtain",
    "screen",
    "monitor",
    "television",
    "tv",
)

ALIASES = {
    "bookcase": {"bookcase", "book shelf", "bookshelf"},
    "cabinet": {"cabinet", "chest of drawers", "case"},
    "ceiling": {"ceiling"},
    "chair": {"chair", "armchair", "swivel chair", "seat"},
    "curtain": {"curtain", "blind", "shutter"},
    "desk": {"desk"},
    "door": {"door"},
    "floor": {"floor", "flooring"},
    "monitor": {"monitor", "screen", "crt screen", "computer screen"},
    "screen": {"screen", "crt screen", "monitor", "computer screen"},
    "shelf": {"shelf", "book shelf", "bookshelf", "bookcase"},
    "sofa": {"sofa", "couch"},
    "table": {"table", "desk", "coffee table", "dining table"},
    "television": {"television", "tv", "screen"},
    "tv": {"television", "tv", "screen"},
    "wall": {"wall"},
    "wardrobe": {"wardrobe", "closet"},
    "window": {"window", "windowpane"},
    "windowpane": {"windowpane", "window"},
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run an ADE20K semantic segmentation model on FastVGGT sampled frames, "
            "filter corresponding 3D points, and export RoomFormer-style density maps."
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
        "--model",
        default="nvidia/segformer-b2-finetuned-ade-512-512",
        help="Hugging Face ADE20K semantic segmentation model.",
    )
    parser.add_argument(
        "--mode",
        choices=["keep-structural", "delete-furniture", "both"],
        default="both",
        help="Which filtered density maps to export.",
    )
    parser.add_argument(
        "--keep-classes",
        nargs="+",
        default=list(DEFAULT_KEEP_CLASSES),
        help="ADE20K classes to keep for structural_density.png.",
    )
    parser.add_argument(
        "--remove-classes",
        nargs="+",
        default=list(DEFAULT_REMOVE_CLASSES),
        help="ADE20K classes to remove for non_furniture_density.png.",
    )
    parser.add_argument(
        "--mask-dilate",
        type=int,
        default=2,
        help="Dilate semantic masks by this many pixels at FastVGGT output resolution.",
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
        help="Optional confidence key used after semantic filtering. Use a missing key such as none to skip.",
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
        help="Maximum filtered points to write per PLY. Use 0 to skip PLY files.",
    )
    parser.add_argument(
        "--save-mask-debug",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save per-frame semantic mask overlays for inspection.",
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


def normalize_label(label):
    return str(label).lower().replace("_", " ").strip()


def requested_label_names(names):
    expanded = set()
    for name in names:
        normalized = normalize_label(name)
        expanded.add(normalized)
        expanded.update(ALIASES.get(normalized, set()))
    return expanded


def ids_for_labels(id2label, requested_names):
    requested = requested_label_names(requested_names)
    matched = {}
    for class_id, label in id2label.items():
        normalized = normalize_label(label)
        if normalized in requested:
            matched[int(class_id)] = normalized
    missing = sorted(set(map(normalize_label, requested_names)) - set(matched.values()))
    return matched, missing


def dilate_mask(mask, radius):
    if radius <= 0:
        return mask
    kernel_size = radius * 2 + 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)


def colorize_mask(mask, color):
    colored = np.zeros((*mask.shape, 3), dtype=np.uint8)
    colored[mask] = color
    return colored


def write_debug(path, frame_path, structural_mask, furniture_mask):
    frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
    if frame is None:
        return
    structural = cv2.resize(
        structural_mask.astype(np.uint8),
        (frame.shape[1], frame.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)
    furniture = cv2.resize(
        furniture_mask.astype(np.uint8),
        (frame.shape[1], frame.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)
    overlay = frame.copy()
    overlay[structural] = (0, 180, 0)
    overlay[furniture] = (0, 0, 255)
    blended = cv2.addWeighted(frame, 0.62, overlay, 0.38, 0.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), blended)


def semantic_masks_for_frames(frame_paths, output_hw, model_name, keep_classes, remove_classes, dilate, debug_dir):
    try:
        import torch
        from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
    except ImportError as exc:
        raise ImportError(
            "ADE20K filtering requires transformers and torch. Install with: "
            'uv pip install --python "$(which python)" transformers pillow'
        ) from exc

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForSemanticSegmentation.from_pretrained(model_name).to(device).eval()
    id2label = {int(k): v for k, v in model.config.id2label.items()}

    keep_ids, missing_keep = ids_for_labels(id2label, keep_classes)
    remove_ids, missing_remove = ids_for_labels(id2label, remove_classes)
    if not keep_ids and not remove_ids:
        raise ValueError("No requested ADE20K classes matched the model labels.")

    print("Keeping structural classes:", ", ".join(f"{name}({idx})" for idx, name in keep_ids.items()) or "none")
    print("Removing furniture classes:", ", ".join(f"{name}({idx})" for idx, name in remove_ids.items()) or "none")
    if missing_keep:
        print("Warning: unmatched keep classes:", ", ".join(missing_keep))
    if missing_remove:
        print("Warning: unmatched remove classes:", ", ".join(missing_remove))

    structural_masks = []
    furniture_masks = []
    labels_per_frame = []
    output_height, output_width = output_hw

    with torch.no_grad():
        for frame_index, frame_path in enumerate(frame_paths):
            image = Image.open(frame_path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt").to(device)
            logits = model(**inputs).logits
            logits = torch.nn.functional.interpolate(
                logits,
                size=(output_height, output_width),
                mode="bilinear",
                align_corners=False,
            )
            labels = logits.argmax(dim=1)[0].detach().cpu().numpy().astype(np.int32)

            structural_mask = np.isin(labels, list(keep_ids.keys()))
            furniture_mask = np.isin(labels, list(remove_ids.keys()))
            structural_mask = dilate_mask(structural_mask, dilate)
            furniture_mask = dilate_mask(furniture_mask, dilate)

            structural_masks.append(structural_mask)
            furniture_masks.append(furniture_mask)
            present_ids = sorted(int(i) for i in np.unique(labels))
            labels_per_frame.append([id2label[i] for i in present_ids if i in id2label])

            if debug_dir is not None:
                write_debug(debug_dir / f"{frame_index:06d}.png", frame_path, structural_mask, furniture_mask)

    return (
        np.stack(structural_masks, axis=0),
        np.stack(furniture_masks, axis=0),
        keep_ids,
        remove_ids,
        missing_keep,
        missing_remove,
        labels_per_frame,
    )


def base_keep_mask(predictions, points_flat, conf_key, conf_thresh):
    keep = np.isfinite(points_flat).all(axis=1)
    if conf_key in predictions:
        conf = np.asarray(predictions[conf_key], dtype=np.float32).reshape(-1)
        if conf.shape[0] == points_flat.shape[0]:
            keep &= np.isfinite(conf) & (conf >= conf_thresh)
    return keep


def export_variant(args, output_name, points_flat, keep_mask, predictions):
    points = points_flat[keep_mask]
    if points.shape[0] == 0:
        print(f"Skipping {output_name}: no points remain after filtering.")
        return None

    density_path = args.output_dir / f"{output_name}.png"
    selected_plane = write_density_outputs(
        density_path,
        points,
        plane=args.plane,
        size=args.size,
        padding=args.padding,
        flip_x=args.flip_x,
        flip_y=args.flip_y,
    )
    print(f"Wrote {density_path} from {points.shape[0]} points (plane={selected_plane}).")

    if args.max_ply_points > 0 and "images" in predictions:
        colors = point_colors(predictions, keep_mask)
        ply_path = args.output_dir / f"{output_name}.ply"
        write_ply(ply_path, points_flat[keep_mask], colors, args.max_ply_points)
        print(f"Wrote {ply_path}")

    return {
        "path": str(density_path),
        "points_kept": int(points.shape[0]),
        "selected_plane": selected_plane,
    }


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

    debug_dir = args.output_dir / "semantic_debug" if args.save_mask_debug else None
    (
        structural_masks,
        furniture_masks,
        keep_ids,
        remove_ids,
        missing_keep,
        missing_remove,
        labels_per_frame,
    ) = semantic_masks_for_frames(
        frame_paths=frame_paths,
        output_hw=(height, width),
        model_name=args.model,
        keep_classes=args.keep_classes,
        remove_classes=args.remove_classes,
        dilate=args.mask_dilate,
        debug_dir=debug_dir,
    )

    points_flat = points_grid.reshape(-1, 3)
    keep_base = base_keep_mask(predictions, points_flat, args.conf_key, args.conf_thresh)
    structural_flat = structural_masks.reshape(-1)
    furniture_flat = furniture_masks.reshape(-1)

    outputs = {}
    if args.mode in {"keep-structural", "both"}:
        outputs["structural_density"] = export_variant(
            args,
            "structural_density",
            points_flat,
            keep_base & structural_flat,
            predictions,
        )
    if args.mode in {"delete-furniture", "both"}:
        outputs["non_furniture_density"] = export_variant(
            args,
            "non_furniture_density",
            points_flat,
            keep_base & ~furniture_flat,
            predictions,
        )

    report = {
        "predictions": str(args.predictions),
        "frames_dir": str(args.frames_dir),
        "model": args.model,
        "mode": args.mode,
        "keep_classes": args.keep_classes,
        "remove_classes": args.remove_classes,
        "matched_keep_classes": keep_ids,
        "matched_remove_classes": remove_ids,
        "missing_keep_classes": missing_keep,
        "missing_remove_classes": missing_remove,
        "num_frames": num_frames,
        "points_total": int(points_flat.shape[0]),
        "points_after_conf": int(keep_base.sum()),
        "points_in_structural_masks": int((keep_base & structural_flat).sum()),
        "points_in_furniture_masks": int((keep_base & furniture_flat).sum()),
        "outputs": outputs,
        "labels_per_frame": labels_per_frame,
    }
    report_path = args.output_dir / "ade20k_filter_report.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
