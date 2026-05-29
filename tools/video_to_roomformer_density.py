#!/usr/bin/env python3
"""Extract video frames, run FastVGGT, and export RoomFormer density input."""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.vggt_npz_to_roomformer_density import load_points, points_to_density


def parse_args():
    parser = argparse.ArgumentParser(
        description="Video -> sampled frames -> VGGT predictions -> RoomFormer density PNG."
    )
    parser.add_argument("--video", type=Path, required=True, help="Input video path.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for frames, predictions, point cloud, and density.",
    )
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=REPO_ROOT / "ckpt" / "model_tracker_fixed_e20.pt",
        help="FastVGGT checkpoint path.",
    )
    parser.add_argument("--sample-fps", type=float, default=1.0, help="Frames to sample per second.")
    parser.add_argument("--max-frames", type=int, default=0, help="Optional maximum sampled frames.")
    parser.add_argument("--merging", type=int, default=0, help="FastVGGT merging block index.")
    parser.add_argument("--merge-ratio", type=float, default=0.9, help="FastVGGT merge ratio.")
    parser.add_argument(
        "--points-key",
        default="world_points_from_depth",
        choices=["world_points_from_depth", "world_points"],
        help="Point set used for density and PLY export.",
    )
    parser.add_argument(
        "--conf-key",
        default="depth_conf",
        help="Confidence key used to filter points.",
    )
    parser.add_argument("--conf-thresh", type=float, default=3.0)
    parser.add_argument("--plane", choices=["xy", "xz", "yz"], default="xz")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--padding", type=float, default=0.05)
    parser.add_argument("--flip-x", action="store_true")
    parser.add_argument("--flip-y", action="store_true")
    parser.add_argument(
        "--max-ply-points",
        type=int,
        default=500000,
        help="Maximum points to write to point_cloud.ply. Use 0 to skip PLY.",
    )
    return parser.parse_args()


def extract_frames(video_path, frames_dir, sample_fps, max_frames):
    if sample_fps <= 0:
        raise ValueError("--sample-fps must be greater than 0")
    frames_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise OSError(f"Failed to open video: {video_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if source_fps <= 0 or not np.isfinite(source_fps):
        source_fps = sample_fps
    interval = max(1, int(round(source_fps / sample_fps)))

    frame_paths = []
    frame_idx = 0
    sampled_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % interval == 0:
            out_path = frames_dir / f"{sampled_idx:06d}.png"
            if not cv2.imwrite(str(out_path), frame):
                raise OSError(f"Failed to write frame: {out_path}")
            frame_paths.append(out_path)
            sampled_idx += 1
            if max_frames > 0 and sampled_idx >= max_frames:
                break
        frame_idx += 1
    cap.release()

    if not frame_paths:
        raise ValueError(f"No frames were extracted from {video_path}")
    return frame_paths, source_fps, interval


def load_model(ckpt_path, args):
    import torch
    from vggt.models.vggt import VGGT

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for FastVGGT inference.")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model = VGGT(
        enable_point=True,
        enable_track=True,
        merging=args.merging,
        merge_ratio=args.merge_ratio,
    )
    state_dict = torch.load(ckpt_path, map_location="cpu")
    incompat = model.load_state_dict(state_dict, strict=False)
    if incompat.missing_keys or incompat.unexpected_keys:
        print(f"Loaded checkpoint with partial key mismatch: {incompat}")
    return model.cuda().eval()


def run_vggt(model, frame_paths):
    import torch
    from vggt.utils.geometry import unproject_depth_map_to_point_map
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    images = load_and_preprocess_images([str(p) for p in frame_paths]).cuda()
    _, _, height, width = images.shape
    model.update_patch_dimensions(width // 14, height // 14)
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            predictions = model(images)

    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic

    for key, value in list(predictions.items()):
        if isinstance(value, torch.Tensor):
            predictions[key] = value.detach().float().cpu().numpy().squeeze(0)
    predictions.pop("pose_enc_list", None)
    predictions["world_points_from_depth"] = unproject_depth_map_to_point_map(
        predictions["depth"], predictions["extrinsic"], predictions["intrinsic"]
    ).astype(np.float32)
    return predictions


def point_colors(predictions, mask):
    images = predictions["images"]
    if images.ndim == 4 and images.shape[1] == 3:
        colors = np.transpose(images, (0, 2, 3, 1))
    else:
        colors = images
    colors = np.clip(colors.reshape(-1, 3) * 255.0, 0, 255).astype(np.uint8)
    return colors[mask]


def write_ply(path, points, colors, max_points):
    if max_points <= 0:
        return
    if points.shape[0] > max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(points.shape[0], size=max_points, replace=False)
        points = points[idx]
        colors = colors[idx]

    with path.open("w", encoding="utf-8") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {points.shape[0]}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        handle.write("end_header\n")
        for point, color in zip(points, colors):
            handle.write(
                f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = args.output_dir / "frames"

    frame_paths, source_fps, interval = extract_frames(
        args.video, frames_dir, args.sample_fps, args.max_frames
    )
    print(
        f"Extracted {len(frame_paths)} frames to {frames_dir} "
        f"(source_fps={source_fps:.3f}, interval={interval})."
    )

    model = load_model(args.ckpt, args)
    predictions = run_vggt(model, frame_paths)

    predictions_path = args.output_dir / "predictions.npz"
    np.savez(predictions_path, **predictions)
    print(f"Wrote {predictions_path}")

    points_all = np.asarray(predictions[args.points_key], dtype=np.float32).reshape(-1, 3)
    mask = np.isfinite(points_all).all(axis=1)
    if args.conf_key in predictions:
        conf = np.asarray(predictions[args.conf_key], dtype=np.float32).reshape(-1)
        if conf.shape[0] == points_all.shape[0]:
            mask &= np.isfinite(conf) & (conf >= args.conf_thresh)

    points = load_points(predictions, args.points_key, args.conf_key, args.conf_thresh)
    density = points_to_density(
        points,
        plane=args.plane,
        size=args.size,
        padding=args.padding,
        flip_x=args.flip_x,
        flip_y=args.flip_y,
    )

    density_path = args.output_dir / "density.png"
    if not cv2.imwrite(str(density_path), (density * 255).astype(np.uint8)):
        raise OSError(f"Failed to write {density_path}")
    print(f"Wrote {density_path} from {points.shape[0]} points")

    if args.max_ply_points > 0:
        colors = point_colors(predictions, mask)
        write_ply(args.output_dir / "point_cloud.ply", points_all[mask], colors, args.max_ply_points)
        print(f"Wrote {args.output_dir / 'point_cloud.ply'}")


if __name__ == "__main__":
    main()
