"""Draw the landmarks of one video frame and the normalized features computed from it.

    python scripts/visualize_landmarks.py --video data/raw/hello/MVI_0029.MOV --out results/figures/landmarks_hello.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config  # noqa: E402
from src.features import FeatureLayout, frame_features  # noqa: E402
from src.landmarks import HAND_CONNECTIONS, HolisticTracker, RawLandmarks, draw_landmarks, iter_video_frames, video_fps  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--frame", type=int, default=None, help="frame index; default is the middle frame with both hands, else any hand")
    parser.add_argument("--out", type=Path, default=Path("results/figures/landmarks_example.png"))
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    layout = FeatureLayout.from_config(cfg)

    images, frames = [], []
    with HolisticTracker(cfg["holistic"]) as tracker:
        for image in iter_video_frames(args.video):
            frames.append(tracker.process(image))
            images.append(image)
    raw = RawLandmarks.from_frames(frames, video_fps(args.video))
    feats = frame_features(raw, layout, cfg)
    flags = feats[:, layout.blocks["FLAGS"]]

    if args.frame is None:
        candidates = np.flatnonzero(flags.all(axis=1))
        if len(candidates) == 0:
            candidates = np.flatnonzero(flags.any(axis=1))
        idx = int(candidates[len(candidates) // 2]) if len(candidates) else len(frames) // 2
    else:
        idx = args.frame

    print(f"video: {args.video} | frames: {len(raw)} | fps: {raw.fps:.1f}")
    print(f"raw shapes: left {raw.left_hand.shape}, right {raw.right_hand.shape}, pose {raw.pose.shape}, face {raw.face.shape}")
    print(f"per-frame features: {feats.shape} | frame {idx} flags (left, right): {flags[idx]}")
    for name, sl in layout.blocks.items():
        print(f"  {name:10s} columns {sl.start:3d}-{sl.stop - 1:3d}  ({sl.stop - sl.start} values)")

    image = images[idx].copy()
    draw_landmarks(image, frames[idx], list(layout.face_indices), list(layout.pose_indices))
    image = cv2.resize(image, (960, round(image.shape[0] * 960 / image.shape[1])))

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2), gridspec_kw={"width_ratios": [1.8, 1, 1]})
    axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f"{args.video.parent.name}/{args.video.name}, frame {idx}")
    axes[0].axis("off")

    ax = axes[1]
    for name, color in (("LEFT_HAND", "tab:blue"), ("RIGHT_HAND", "tab:orange")):
        pts = feats[idx, layout.blocks[name]].reshape(-1, 3)
        if not pts.any():
            continue
        for a, b in HAND_CONNECTIONS:
            ax.plot(pts[[a, b], 0], pts[[a, b], 1], color=color, lw=1.5)
        ax.scatter(pts[:, 0], pts[:, 1], s=12, color=color, label=name.lower().replace("_", " "))
    ax.scatter([0], [0], marker="x", color="black", label="wrist = origin")
    ax.set_title("Hands: wrist-centred, unit = wrist to middle MCP")
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[2]
    pose = feats[idx, layout.blocks["POSE"]].reshape(-1, 3)
    face = feats[idx, layout.blocks["FACE"]].reshape(-1, 3)
    ax.scatter(face[:, 0], face[:, 1], s=6, color="tab:red", label="face subset")
    ax.scatter(pose[:, 0], pose[:, 1], s=30, color="tab:green", label="pose subset")
    ax.scatter([0], [0], marker="x", color="black", label="shoulder midpoint = origin")
    ax.set_title("Pose and face: unit = shoulder width")
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=110)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
