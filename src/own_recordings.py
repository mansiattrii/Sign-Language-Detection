"""File naming and writing for own recordings: data/own_recordings/<word>/<signer>_<n>.mp4."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def next_take_path(folder: Path, signer: str) -> Path:
    """The lowest unused <signer>_<n>.mp4 in folder, so a deleted take's number is reused."""
    taken = {p.stem.rsplit("_", 1)[-1] for p in folder.glob(f"{signer}_*.mp4")}
    n = 1
    while str(n) in taken:
        n += 1
    return folder / f"{signer}_{n}.mp4"


def write_video(path: Path, frames: list[np.ndarray], fps: float) -> None:
    """Write BGR frames as an mp4 (unmirrored, exactly as the camera delivered them)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()
