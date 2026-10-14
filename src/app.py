"""Live ISL word recognition from a webcam, building a sentence from recognized words.

    python -m src.app
    python -m src.app --practice --signer mansi --takes 5
    python -m src.app --practice --signer mansi --words hello thank_you --takes 10
    python -m src.app --video data/raw/hello/MVI_0029.MOV --headless

Keys: c clears the sentence, Backspace deletes the last word, q quits.
Practice mode: the app asks for a word; each sign you make is predicted, shown and saved as
data/own_recordings/<WORD>/<signer>_<n>.mp4. r deletes the last saved take, n skips to the next word.
"""
from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.config import load_config
from src.landmarks import FrameLandmarks, HolisticTracker, draw_landmarks, video_fps
from src.recognizer import Prediction, Recognizer
from src.segmenter import Segment, Segmenter, State, hand_motion

KEY_BACKSPACE = 8
DISPLAY_SIZE = (960, 720)
STATE_COLORS = {State.IDLE: (150, 150, 150), State.SIGNING: (60, 200, 60), State.END: (0, 160, 255)}
WHITE, BLACK, GREEN, RED = (255, 255, 255), (0, 0, 0), (60, 200, 60), (60, 60, 230)


@dataclass
class Take:
    path: Path
    word: str
    prediction: Prediction | None
    time: str


class Practice:
    """Prompts words in turn and saves every segmented sign as a labelled clip."""

    LOG_FIELDS = ["time", "word", "file", "predicted", "confidence", "top3"]

    def __init__(self, words: list[str], takes: int, signer: str, own_dir: Path):
        self.words, self.takes, self.signer, self.own_dir = words, takes, signer, own_dir
        self.index = 0
        self.saved: list[Take] = []
        self.log_path = own_dir / "practice_logs" / f"{time.strftime('%Y%m%d_%H%M%S')}_{signer}.csv"

    @property
    def done(self) -> bool:
        return self.index >= len(self.words)

    @property
    def target(self) -> str:
        return self.words[self.index]

    @property
    def count(self) -> int:
        return sum(t.word == self.target for t in self.saved)

    def _next_path(self) -> Path:
        folder = self.own_dir / self.target
        folder.mkdir(parents=True, exist_ok=True)
        taken = {p.stem.rsplit("_", 1)[-1] for p in folder.glob(f"{self.signer}_*.mp4")}
        n = 1
        while str(n) in taken:
            n += 1
        return folder / f"{self.signer}_{n}.mp4"

    def save(self, images: list[np.ndarray], fps: float, prediction: Prediction | None) -> Take:
        path = self._next_path()
        h, w = images[0].shape[:2]
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for image in images:
            writer.write(image)
        writer.release()
        take = Take(path, self.target, prediction, time.strftime("%Y-%m-%d %H:%M:%S"))
        self.saved.append(take)
        self._write_log()
        if self.count >= self.takes:
            self.index += 1
        return take

    def undo(self) -> Take | None:
        """Delete the most recent take and ask for that word again."""
        if not self.saved:
            return None
        take = self.saved.pop()
        take.path.unlink(missing_ok=True)
        self.index = self.words.index(take.word)
        self._write_log()
        return take

    def skip(self) -> None:
        self.index += 1

    def _write_log(self) -> None:
        """Rewrite this session's log, one file per session, so an undone take leaves no row."""
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.LOG_FIELDS)
            writer.writeheader()
            for t in self.saved:
                p = t.prediction
                writer.writerow({
                    "time": t.time, "word": t.word,
                    "file": t.path.relative_to(self.own_dir).as_posix(),
                    "predicted": p.word if p else "", "confidence": f"{p.confidence:.3f}" if p else "",
                    "top3": " ".join(w for w, _ in p.top) if p else "",
                })


class LiveApp:
    """Everything except the camera and the window, so it also runs on a video file."""

    def __init__(self, cfg: dict[str, Any], recognizer: Recognizer, practice: Practice | None = None,
                 debug: bool = False):
        self.cfg = cfg
        self.recognizer = recognizer
        self.practice = practice
        self.debug = debug
        self.segmenter = Segmenter(cfg["segmenter"])
        self.motion_threshold = cfg["segmenter"]["motion_threshold"]
        self.min_confidence = cfg["live"]["min_confidence"]
        groups = recognizer.bundle["groups"]
        self.face_indices = recognizer.layout.face_indices if "FACE" in groups else []
        self.pose_indices = recognizer.layout.pose_indices if "POSE" in groups else []
        self.sentence: list[str] = []
        self.last: Prediction | None = None
        self.message = ""
        self.prev: FrameLandmarks | None = None
        self.motion = 0.0
        self.fps = 0.0
        self._last_time: float | None = None

    def tick_fps(self, now: float) -> None:
        if self._last_time is not None and now > self._last_time:
            inst = 1.0 / (now - self._last_time)
            self.fps = inst if self.fps == 0 else 0.9 * self.fps + 0.1 * inst
        self._last_time = now

    def step(self, frame: np.ndarray, landmarks: FrameLandmarks, fps: float) -> Prediction | None:
        """Advance by one frame; returns a prediction on the frame where a sign ends."""
        present = landmarks.left_hand is not None or landmarks.right_hand is not None
        self.motion = hand_motion(self.prev, landmarks)
        self.prev = landmarks
        item = (landmarks, frame if self.practice else None)
        segment = self.segmenter.update(item, present, self.motion > self.motion_threshold)
        return self._finish(segment, fps) if segment is not None else None

    def _finish(self, segment: Segment, fps: float) -> Prediction | None:
        prediction = self.recognizer.predict([lm for lm, _ in segment.items], fps)
        if self.practice is not None and not self.practice.done:
            if prediction is None:
                self.message = "too short, not saved - sign again"
                return None
            target = self.practice.target
            take = self.practice.save([img for _, img in segment.items], fps, prediction)
            verdict = "OK" if prediction.word == target else "WRONG"
            self.message = f"saved {take.path.name}: {_pretty(prediction.word)} {prediction.confidence:.2f} {verdict}"
        if prediction is None:
            self.message = "too short to recognize"
            return None
        self.last = prediction
        if self.practice is None:
            if prediction.confidence >= self.min_confidence:
                self.sentence.append(prediction.word)
                self.message = ""
            else:
                self.message = f"not sure ({prediction.confidence:.2f}), not added"
        return prediction

    def handle_key(self, key: int) -> bool:
        """Returns False when the app should quit."""
        if key == ord("q"):
            return False
        if key == ord("c"):
            self.sentence.clear()
        elif key == KEY_BACKSPACE and self.sentence:
            self.sentence.pop()
        elif self.practice is not None and key == ord("r"):
            take = self.practice.undo()
            self.message = f"deleted {take.path.name}, sign {_pretty(take.word)} again" if take else "nothing to undo"
        elif self.practice is not None and key == ord("n") and not self.practice.done:
            self.practice.skip()
            self.message = "skipped"
        return True

    def render(self, frame: np.ndarray, landmarks: FrameLandmarks) -> np.ndarray:
        image = draw_landmarks(frame.copy(), landmarks, self.face_indices, self.pose_indices)
        # Mirror only for display; Holistic always sees the unmirrored frame.
        image = cv2.resize(cv2.flip(image, 1), DISPLAY_SIZE)
        w, h = DISPLAY_SIZE

        state = self.segmenter.state
        _label(image, state.value, (16, 40), STATE_COLORS[state], scale=0.9)
        _label(image, f"{self.fps:4.1f} FPS", (16, 80), WHITE, scale=0.6)
        if self.debug:
            _label(image, f"motion {min(self.motion, 9.99):.3f} (threshold {self.motion_threshold})",
                   (16, 112), WHITE, scale=0.55)

        if self.last is not None:
            x = w - 340
            color = GREEN if self.last.confidence >= self.min_confidence else WHITE
            _label(image, f"{_pretty(self.last.word)} {self.last.confidence:.2f}", (x, 40), color, scale=0.9)
            for i, (word, p) in enumerate(self.last.top):
                y = 80 + 30 * i
                cv2.rectangle(image, (x, y - 16), (x + int(100 * p), y + 4), (200, 140, 40), -1)
                _label(image, f"{_pretty(word)} {p:.2f}", (x + 108, y), WHITE, scale=0.55)

        if self.practice is not None:
            if self.practice.done:
                prompt = "practice complete - press q"
            else:
                prompt = (f"Sign: {_pretty(self.practice.target)}   take {self.practice.count + 1}/{self.practice.takes}"
                          f"   ({self.practice.index + 1}/{len(self.practice.words)} words)")
            _label(image, prompt, (16, h - 100), (0, 220, 255), scale=0.8)
            keys = "r redo last   n next word   q quit"
        else:
            sentence = " ".join(_pretty(word) for word in self.sentence)
            _label(image, _fit(sentence or "...", 50), (16, h - 60), WHITE, scale=0.9)
            keys = "c clear   Backspace delete   q quit"
        if self.message:
            _label(image, self.message, (16, h - 140 if self.practice else h - 100), WHITE, scale=0.6)
        _label(image, keys, (16, h - 20), (200, 200, 200), scale=0.5)
        return image


def _pretty(word: str) -> str:
    return word.replace("_", " ").upper()


def _fit(text: str, max_chars: int) -> str:
    return text if len(text) <= max_chars else "..." + text[-(max_chars - 3):]


def _label(image: np.ndarray, text: str, org: tuple[int, int], color: tuple[int, int, int], scale: float) -> None:
    """Text with a dark outline, readable on any background."""
    cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, BLACK, 4, cv2.LINE_AA)
    cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1 if scale < 0.7 else 2, cv2.LINE_AA)


def open_camera(live_cfg: dict[str, Any]) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(live_cfg["camera_index"], cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, live_cfg["frame_width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, live_cfg["frame_height"])
    if not cap.isOpened():
        raise IOError(f"cannot open camera {live_cfg['camera_index']}")
    return cap


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--model", type=Path, default=None, help="default from config (live.model)")
    parser.add_argument("--camera", type=int, default=None, help="camera index, default from config")
    parser.add_argument("--video", type=Path, default=None, help="run on a video file instead of the webcam")
    parser.add_argument("--headless", action="store_true", help="no window; print predictions (with --video)")
    parser.add_argument("--debug", action="store_true", help="show the motion value used by the segmenter")
    parser.add_argument("--practice", action="store_true", help="prompt words and save every take")
    parser.add_argument("--signer", default=None, help="practice mode: name used in saved file names")
    parser.add_argument("--takes", type=int, default=5, help="practice mode: takes per word")
    parser.add_argument("--words", nargs="+", default=None, help="practice mode: words to practise, default all")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.camera is not None:
        cfg["live"]["camera_index"] = args.camera
    recognizer = Recognizer.from_path(args.model or Path(cfg["live"]["model"]), cfg)

    practice = None
    if args.practice:
        if not args.signer:
            parser.error("--practice needs --signer")
        words = args.words or cfg["words"]
        unknown = set(words) - set(cfg["words"])
        if unknown:
            parser.error(f"not in the vocabulary: {sorted(unknown)}")
        practice = Practice(words, args.takes, args.signer, Path(cfg["paths"]["own_dir"]))

    app = LiveApp(cfg, recognizer, practice, args.debug)
    cap = cv2.VideoCapture(str(args.video)) if args.video else open_camera(cfg["live"])
    file_fps = video_fps(args.video) if args.video else None
    window = "ISL recognition"
    try:
        with HolisticTracker(cfg["holistic"]) as tracker:
            t = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                now = t / file_fps if file_fps else time.perf_counter()
                app.tick_fps(now)
                landmarks = tracker.process(frame)
                prediction = app.step(frame, landmarks, file_fps or max(app.fps, 1.0))
                t += 1
                if args.headless:
                    if prediction is not None:
                        top = ", ".join(f"{w} {p:.2f}" for w, p in prediction.top)
                        print(f"frame {t}: {top}")
                    continue
                cv2.imshow(window, app.render(frame, landmarks))
                key = cv2.waitKey(1) & 0xFF
                if not app.handle_key(key) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    break
            if args.headless and args.video:
                # A file can end mid-sign; flush it as if the hands had left the view.
                for _ in range(cfg["segmenter"]["end_absent_frames"]):
                    prediction = app.step(frame, FrameLandmarks(None, None, None, None), file_fps)
                    if prediction is not None:
                        print("end of video: " + ", ".join(f"{w} {p:.2f}" for w, p in prediction.top))
            if app.sentence:
                print("sentence:", " ".join(_pretty(w) for w in app.sentence))
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
