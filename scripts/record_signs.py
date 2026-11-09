"""Record fixed-length takes of one word for the own-recordings test set.

    python scripts/record_signs.py --word hello --signer mansi --count 20

Space starts a take: a countdown, then `--seconds` of recording, saved as
data/own_recordings/<word>/<signer>_<n>.mp4. Start each take with your hands down and lower them
again when the sign is done, like the INCLUDE signers. r deletes the last take of this session so
it can be redone, q quits.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.app import _label, _pretty, open_camera  # noqa: E402
from src.config import load_config  # noqa: E402
from src.own_recordings import next_take_path, write_video  # noqa: E402

YELLOW, WHITE, RED = (0, 220, 255), (255, 255, 255), (60, 60, 230)


def show(window: str, frame: np.ndarray, lines: list[tuple[str, tuple[int, int, int], float]],
         big: str | None = None) -> int:
    """Mirror the frame for display, draw the text, and return the pressed key."""
    image = cv2.flip(frame, 1)
    for i, (text, color, scale) in enumerate(lines):
        _label(image, text, (16, 36 + 34 * i), color, scale)
    if big:
        h, w = image.shape[:2]
        _label(image, big, (w // 2 - 30, h // 2), YELLOW, 3.0)
    cv2.imshow(window, image)
    return cv2.waitKey(1) & 0xFF


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--word", required=True)
    parser.add_argument("--signer", required=True)
    parser.add_argument("--count", type=int, default=20, help="takes to record in this session")
    parser.add_argument("--seconds", type=float, default=3.0, help="length of each take")
    parser.add_argument("--countdown", type=int, default=3)
    parser.add_argument("--camera", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    word = args.word.lower()
    if word not in cfg["words"]:
        parser.error(f"'{word}' is not in the vocabulary: {', '.join(cfg['words'])}")
    if args.camera is not None:
        cfg["live"]["camera_index"] = args.camera
    folder = Path(cfg["paths"]["own_dir"]) / word

    cap = open_camera(cfg["live"])
    window = "Record signs"
    saved: list[Path] = []
    message = ""
    try:
        while len(saved) < args.count:
            ok, frame = cap.read()
            if not ok:
                raise IOError("camera stopped delivering frames")
            key = show(window, frame, [
                (f"{_pretty(word)}: {len(saved)}/{args.count} takes", YELLOW, 0.9),
                ("space record   r redo last   q quit", WHITE, 0.6),
                (message, WHITE, 0.6),
            ])
            if key == ord("q"):
                break
            if key == ord("r"):
                if saved:
                    last = saved.pop()
                    last.unlink(missing_ok=True)
                    message = f"deleted {last.name}"
                else:
                    message = "nothing to redo in this session"
                continue
            if key != ord(" "):
                continue

            end = time.perf_counter() + args.countdown
            while (left := end - time.perf_counter()) > 0:
                ok, frame = cap.read()
                if not ok:
                    break
                show(window, frame, [(f"{_pretty(word)}: get ready, hands down", YELLOW, 0.9)], str(int(left) + 1))

            frames: list[np.ndarray] = []
            start = time.perf_counter()
            while time.perf_counter() - start < args.seconds:
                ok, frame = cap.read()
                if not ok:
                    break
                frames.append(frame)
                show(window, frame, [(f"RECORDING {_pretty(word)}", RED, 0.9)])
            fps = len(frames) / max(time.perf_counter() - start, 1e-6)
            if len(frames) < 10:
                message = "too few frames captured, take not saved"
                continue
            path = next_take_path(folder, args.signer)
            write_video(path, frames, fps)
            saved.append(path)
            message = f"saved {path.name} ({len(frames)} frames, {fps:.1f} fps)"
            print(message)
    finally:
        cap.release()
        cv2.destroyAllWindows()
    print(f"{len(saved)} take(s) of '{word}' saved in {folder}")


if __name__ == "__main__":
    main()
