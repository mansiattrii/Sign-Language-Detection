"""Measure the live pipeline (segmenter + recognizer) without a webcam.

Streams the cached landmarks of the split's clips back to back, with a pause between signs, through
the same Segmenter and Recognizer the app uses, and checks whether each sign comes out as exactly
one segment with the right word.

    python scripts/replay_segmenter.py                       # test split, hands dropped between signs
    python scripts/replay_segmenter.py --pause still         # hands held still between signs
    python scripts/replay_segmenter.py --split val --pause-frames 10

Pauses: "absent" inserts frames without hands (the signer lowers their hands); "still" repeats the
last frame of the sign (the signer freezes, hands in view).
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from src.config import load_config
from src.landmarks import FrameLandmarks, RawLandmarks
from src.recognizer import Recognizer
from src.segmenter import Segmenter, hand_motion

PAUSE = -1


def build_stream(paths: list[str], cache: Path, pause: str, pause_frames: int,
                 seed: int) -> tuple[list[tuple[int, FrameLandmarks]], list[int]]:
    """Frames tagged with the index of the clip they come from (PAUSE for inserted frames)."""
    order = np.random.default_rng(seed).permutation(len(paths))
    empty = FrameLandmarks(None, None, None, None)
    stream: list[tuple[int, FrameLandmarks]] = [(PAUSE, empty)] * pause_frames
    for i in order:
        p = Path(paths[i])
        frames = RawLandmarks.load(cache / p.parent.name / f"{p.stem}.npz").frames()
        stream += [(int(i), f) for f in frames]
        filler = empty if pause == "absent" else frames[-1]
        stream += [(PAUSE, filler)] * pause_frames
    return stream, [int(i) for i in order]


def replay(stream: list[tuple[int, FrameLandmarks]], order: list[int], labels: list[str],
           recognizer: Recognizer, seg_cfg: dict, fps: float) -> dict[str, float]:
    segmenter = Segmenter(seg_cfg)
    prev = None
    segments_per_clip: Counter[int] = Counter()
    predicted: dict[int, str] = {}
    spurious = too_short = 0
    for clip, lm in stream:
        present = lm.left_hand is not None or lm.right_hand is not None
        motion = hand_motion(prev, lm)
        prev = lm
        seg = segmenter.update((clip, lm), present, motion > seg_cfg["motion_threshold"])
        if seg is None:
            continue
        owners = Counter(c for c, _ in seg.items if c != PAUSE)
        if not owners:
            spurious += 1
            continue
        owner = owners.most_common(1)[0][0]
        segments_per_clip[owner] += 1
        pred = recognizer.predict([f for _, f in seg.items], fps)
        if pred is None:
            too_short += 1
        elif owner not in predicted:
            predicted[owner] = pred.word

    return {
        "signs": len(order),
        "one_segment": sum(segments_per_clip[i] == 1 for i in order),
        "split": sum(segments_per_clip[i] > 1 for i in order),
        "missed": sum(segments_per_clip[i] == 0 for i in order),
        "spurious": spurious,
        "too_short": too_short,
        "word_accuracy": sum(segments_per_clip[i] == 1 and predicted.get(i) == labels[i] for i in order) / len(order),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--pause", default="absent", choices=["absent", "still"])
    parser.add_argument("--pause-frames", type=int, default=20)
    parser.add_argument("--fps", type=float, default=25.0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    recognizer = Recognizer.from_path(args.model or Path(cfg["live"]["model"]), cfg)
    processed = Path(cfg["paths"]["processed_dir"])
    split = json.loads((processed / "split.json").read_text(encoding="utf-8"))
    paths = split[args.split]["paths"]
    labels = [Path(p).parent.name for p in paths]
    stream, order = build_stream(paths, processed / "landmarks" / "include", args.pause, args.pause_frames,
                                 cfg["seed"])
    r = replay(stream, order, labels, recognizer, cfg["segmenter"], args.fps)

    n = r["signs"]
    print(f"{args.split} split, {n} signs, pause '{args.pause}' x {args.pause_frames} frames")
    print(f"  exactly one segment : {r['one_segment']} ({r['one_segment'] / n:.1%})")
    print(f"  split into several  : {r['split']}")
    print(f"  missed              : {r['missed']}")
    print(f"  segments with no sign frames: {r['spurious']}; segments too short to classify: {r['too_short']}")
    print(f"  word accuracy (one segment and the right word): {r['word_accuracy']:.3f}")


if __name__ == "__main__":
    main()
