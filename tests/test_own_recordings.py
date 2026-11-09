import os

import cv2
import numpy as np

from src.extract import needs_extraction, signer_of
from src.own_recordings import next_take_path, write_video


def test_next_take_path_fills_gaps_and_ignores_other_signers(tmp_path):
    for name in ("mansi_1.mp4", "mansi_3.mp4", "ravi_2.mp4"):
        (tmp_path / name).write_bytes(b"")
    assert next_take_path(tmp_path, "mansi").name == "mansi_2.mp4"
    assert next_take_path(tmp_path, "ravi").name == "ravi_1.mp4"
    assert next_take_path(tmp_path / "missing", "mansi").name == "mansi_1.mp4"


def test_signer_name_round_trips_through_the_file_name(tmp_path):
    path = next_take_path(tmp_path, "mansi_a")
    assert signer_of("own", path) == "mansi_a"


def test_written_video_can_be_read_back(tmp_path):
    frames = [np.full((48, 64, 3), 10 * i, np.uint8) for i in range(12)]
    path = tmp_path / "hello" / "mansi_1.mp4"
    write_video(path, frames, 25.0)
    cap = cv2.VideoCapture(str(path))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 12
    assert cap.get(cv2.CAP_PROP_FPS) == 25.0
    cap.release()


def test_replaced_video_is_extracted_again(tmp_path):
    video, cached = tmp_path / "mansi_1.mp4", tmp_path / "mansi_1.npz"
    video.write_bytes(b"take")
    assert needs_extraction(video, cached)
    cached.write_bytes(b"cache")
    os.utime(video, (1000, 1000))
    os.utime(cached, (2000, 2000))
    assert not needs_extraction(video, cached)
    os.utime(video, (3000, 3000))  # take redone after it was cached
    assert needs_extraction(video, cached)
