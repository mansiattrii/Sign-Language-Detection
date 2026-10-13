"""Turn sign videos into a dataset of fixed-length feature clips.

Stage 1 (slow): run Holistic on each video and cache its raw landmarks under
    data/processed/landmarks/<source>/<word>/<video>.npz
Videos with a cache file are skipped, so the stage can be interrupted and resumed.

Stage 2 (fast): turn every cached clip into a (seq_len, F) array with build_sequence()
and write data/processed/<source>.npz. Changing the feature definition only requires
rerunning this stage.

    python -m src.extract --limit 2      # quick check on 2 videos per word
    python -m src.extract                # everything
    python -m src.extract --source own   # own recordings
"""
from __future__ import annotations

import argparse
import logging
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from src.config import load_config
from src.features import FeatureLayout, build_sequence, shoulder_valid
from src.landmarks import RawLandmarks, extract_video

log = logging.getLogger("extract")


def list_videos(root: Path, words: list[str], extensions: list[str], limit: int | None) -> list[tuple[str, Path]]:
    """(word, path) pairs for every video under root/<word>/, in a stable order."""
    exts = {e.lower() for e in extensions}
    videos = []
    for word in words:
        folder = root / word
        if not folder.is_dir():
            log.warning("No folder for word '%s' in %s", word, root)
            continue
        files = sorted(p for p in folder.iterdir() if p.suffix.lower() in exts)
        videos.extend((word, p) for p in files[:limit])
    return videos


def signer_of(source: str, video: Path) -> str:
    """Own recordings are named <signer>_<n>.mp4; INCLUDE file names carry no signer."""
    return video.stem.rsplit("_", 1)[0] if source == "own" else ""


def cache_path(cache_root: Path, word: str, video: Path) -> Path:
    return cache_root / word / f"{video.stem}.npz"


def extract_landmarks(videos: list[tuple[str, Path]], cache_root: Path, holistic_cfg: dict[str, Any]) -> list[Path]:
    todo = [(w, v) for w, v in videos if not cache_path(cache_root, w, v).exists()]
    log.info("%d videos, %d already cached, %d to process", len(videos), len(videos) - len(todo), len(todo))
    failed = []
    start = time.perf_counter()
    for word, video in tqdm(todo, desc="Holistic", unit="video"):
        try:
            extract_video(video, holistic_cfg).save(cache_path(cache_root, word, video))
        except Exception as exc:  # one corrupt file should not stop a long run
            log.error("Failed on %s: %s", video, exc)
            failed.append(video)
    if todo:
        log.info("Stage 1 took %.1f s (%.2f s per video)", time.perf_counter() - start,
                 (time.perf_counter() - start) / len(todo))
    return failed


def build_dataset(videos: list[tuple[str, Path]], source: str, cache_root: Path, cfg: dict[str, Any],
                  out_path: Path) -> None:
    layout = FeatureLayout.from_config(cfg)
    word_to_id = {w: i for i, w in enumerate(cfg["words"])}
    feats_cfg = cfg["features"]

    clips, labels, paths, signers, n_frames = [], [], [], [], []
    skipped, no_shoulders = Counter(), Counter()
    for word, video in tqdm(videos, desc="Features", unit="clip"):
        cached = cache_path(cache_root, word, video)
        if not cached.exists():
            continue
        raw = RawLandmarks.load(cached)
        if not shoulder_valid(raw.pose, feats_cfg["shoulder_min_visibility"], feats_cfg["eps"]).any():
            no_shoulders[word] += 1
        seq = build_sequence(raw, layout, cfg)
        if seq is None:
            skipped[word] += 1
            continue
        clips.append(seq)
        labels.append(word_to_id[word])
        paths.append(video.as_posix())
        signers.append(signer_of(source, video))
        n_frames.append(len(raw))

    for word in cfg["words"]:
        if skipped[word]:
            log.info("Skipped %d short clip(s) for '%s'", skipped[word], word)
        if no_shoulders[word]:
            log.info("%d clip(s) of '%s' had no usable shoulders (nose fallback)", no_shoulders[word], word)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    x = np.stack(clips) if clips else np.zeros((0, cfg["temporal"]["seq_len"], layout.size), np.float32)
    np.savez_compressed(out_path, X=x, y=np.asarray(labels, dtype=np.int64), words=np.asarray(cfg["words"]),
                        paths=np.asarray(paths), signers=np.asarray(signers), n_frames=np.asarray(n_frames))
    log.info("Wrote %s: X %s, %d clips kept, %d skipped", out_path, x.shape, len(clips), sum(skipped.values()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--source", choices=["include", "own"], default="include")
    parser.add_argument("--limit", type=int, default=None, help="process at most N videos per word")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_config(args.config)
    paths = cfg["paths"]
    root = Path(paths["raw_dir"] if args.source == "include" else paths["own_dir"])
    processed = Path(paths["processed_dir"])
    cache_root = processed / "landmarks" / args.source

    videos = list_videos(root, cfg["words"], cfg["video_extensions"], args.limit)
    failed = extract_landmarks(videos, cache_root, cfg["holistic"])
    suffix = f"_limit{args.limit}" if args.limit else ""
    build_dataset(videos, args.source, cache_root, cfg, processed / f"{args.source}{suffix}.npz")
    if failed:
        log.warning("%d video(s) failed: %s", len(failed), [str(f) for f in failed])


if __name__ == "__main__":
    main()
