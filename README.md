# Real-time Indian Sign Language word recognition

[![tests](https://github.com/mansiattrii/Sign-Language-Detection/actions/workflows/tests.yml/badge.svg)](https://github.com/mansiattrii/Sign-Language-Detection/actions/workflows/tests.yml)

Recognizes 30 isolated Indian Sign Language words from a webcam and builds a running sentence on screen.
Trained on the INCLUDE dataset (620 clips), served as a REST API in Docker.

<!-- demo GIF: results/figures/demo.gif -->
![demo](results/figures/demo.gif)

## What it does

- **Landmarks.** MediaPipe Holistic finds hand, upper-body and face points in every frame.
- **Normalize.** Hands are centred on the wrist and scaled by hand length, body points by shoulder width.
  Missing hands become zeros plus a presence flag, and every clip is resampled to 30 frames.
- **LSTM.** A two-layer LSTM over the 30-frame sequence picks one of 30 words (hands + pose, 149 features).
- **Segmenter.** Live, a small state machine on hand motion decides where each sign starts and ends,
  so you can sign continuously.

## Results

Test split: 136 INCLUDE clips from recording sittings never seen in training. Top-1 accuracy,
mean ± std over 3 seeds. Full tables with top-5 and macro F1 are in `results/ablation*.md`.

| Features | LSTM | GRU | Random Forest |
|---|---|---|---|
| hands | 0.752 ± 0.021 | 0.694 ± 0.011 | 0.591 ± 0.008 |
| **hands + pose** (final) | **0.757 ± 0.060** | 0.679 ± 0.030 | 0.681 ± 0.011 |
| hands + pose + face | 0.691 ± 0.072 | 0.676 ± 0.000 | 0.593 ± 0.011 |

Adding the 159 face columns makes the LSTM worse: with 382 training clips they mostly describe who
is signing rather than what is signed, and the model fits the training sittings instead.
The Random Forest uses per-clip mean and standard deviation, so the 16-point gap to the LSTM is the
value of temporal order. With automatic segmentation instead of pre-cut clips, word accuracy on the
test split drops to 0.625–0.647 (`scripts/replay_segmenter.py`), and 92.6% of signs come out as exactly one segment.

The committed model is the seed-42 run of the final configuration, the weakest of its three seeds:
0.706 top-1, 0.956 top-5 on test (`results/lstm/include/`). The number to quote is the 3-seed mean.

## A bug worth knowing about

The first split kept each word's recording sessions together but ignored that one person signs
many words in a sitting. The confusion matrix gave it away: every GOOD test clip was predicted as
BAD and every BAD as GOOD, the same for I/YOU and GOOD MORNING/GOOD NIGHT. Real confusion is
partial; a clean swap means the model had seen the same person signing the other word in training.
The fix groups clips into 63 sittings by camera file number across all words and sends whole
sittings to one split. Accuracy went down, from an inflated number to an honest 0.757.

## Limitations

- The test set is sitting-independent, not signer-independent. INCLUDE has no signer IDs, so the same
  person can appear in train and test on different days. Expect lower accuracy on a new signer.
- The new-signer test (recording your own clips) was not run. The tooling is in place, so a future
  user can run it in three commands:
  ```powershell
  python scripts/record_signs.py --word hello --signer <name> --count 5   # or: python -m src.app --practice --signer <name> --takes 5
  python -m src.extract --source own
  python -m src.evaluate --model models/lstm.joblib                      # -> results/lstm/own/, with INCLUDE accuracy on the same words
  ```
- Some errors look like sign variants, not model errors. The Random Forest got every HOT clip as
  SCHOOL and every MARKET clip as FATHER; the LSTM still gets 2 of 3 HOT clips as SCHOOL. Total,
  one-directional errors like these fit a signer using a regional variant of the sign.
- No verbs in the vocabulary, so sentences come out telegraphic ("I SICK HOSPITAL").
- 136 test clips is small: one clip is 0.7 points, and seeds differ by up to 14 points.
- The segmenter's frame counts assume about 25 fps, the INCLUDE frame rate; they have not been tuned on a webcam.

## Run it

**API in Docker** (health, predict from a normalized (30, 308) sequence, predict from raw landmarks):

```powershell
docker build -t isl-recognizer .
docker run -p 8000:8000 isl-recognizer
curl localhost:8000/health
# {"status":"ok","model":"lstm.joblib","num_classes":30,"seq_len":30,"num_features":308}
```

`POST /predict` takes `{"sequence": [[...] x 30]}` and `POST /predict/landmarks` takes
`{"fps": 25, "frames": [{"left_hand": [[x,y,z] x 21] | null, "right_hand": ..., "pose": [[x,y,z,v] x 33] | null}, ...]}`
(x and z multiplied by the image aspect ratio, as `src/landmarks.py` stores them). Both return the
top-3 words with probabilities; a wrong shape returns 422 with the expected shape in the message.
Interactive docs at `localhost:8000/docs`.

**Webcam app** (Windows, Python 3.11):

```powershell
py -3.11 -m venv .venv; .\.venv\Scripts\Activate.ps1; python -m pip install -r requirements.txt
python -m src.app               # c clears the sentence, Backspace deletes a word, q quits
python -m src.app --debug       # also shows the motion value the segmenter uses
```

**Tests**, and the segmentation replay for people without a webcam:

```powershell
python -m pytest                        # 76 tests; 4 need the extracted dataset and are skipped without it
python scripts/replay_segmenter.py      # segmenter + model on the INCLUDE test split
python -m src.app --video data/raw/hello/MVI_0029.MOV --headless
```

## Reproduce from scratch

Only the final LSTM (`models/lstm.joblib` + `models/lstm.keras`) and the fixed split
(`data/processed/split.json`) are in git. To rebuild the data cache and retrain:

1. Download INCLUDE from Zenodo record 4010759 (https://zenodo.org/records/4010759). It is split
   into per-category archives laid out as `<Category>/<N>. <Word>/MVI_xxxx.MOV`.
2. Copy the clips for the 30 words in `config.yaml` into `data/raw/<word>/`, keeping the original
   file names, e.g. `data/raw/hello/MVI_0029.MOV`. The split is checked against the exact 620
   paths it was made from; this prints them:
   ```powershell
   python -c "import json; s=json.load(open('data/processed/split.json')); print(*sorted(p for k in ('train','val','test') for p in s[k]['paths']), sep='\n')"
   ```
3. Run the pipeline in order:
   ```powershell
   python -m src.extract                              # ~2.5 s per video on CPU, about 25 min for 620 clips
   python -m src.train --model lstm                   # seed 42 -> models/lstm.joblib + models/lstm.keras
   python -m src.evaluate --model models/lstm.joblib  # -> results/lstm/
   python -m src.ablation                             # 3 feature sets x 3 seeds -> results/ablation.md (--model gru | rf)
   ```
   `python -m src.dataset` only makes a new split; the committed one is reused as is. Raw landmarks
   are cached per video under `data/processed/landmarks/`, so changing features skips MediaPipe.

## Repo layout

| Path | What it holds |
|---|---|
| `src/` | Pipeline: `landmarks` (Holistic), `features` (normalization), `dataset` (sitting split), `augment`, `models`, `train`, `evaluate`, `ablation`, `segmenter`, `predict`, `recognizer`, `app` (webcam) |
| `api/` | FastAPI service around `src.predict` |
| `scripts/` | Segmentation replay, own-clip recording, landmark visualization |
| `tests/` | 76 pytest tests, run in CI |
| `models/` | Final LSTM (seed 42, hands + pose) |
| `data/` | Raw videos and caches (ignored); `split.json` (committed) |
| `results/` | Metrics, confusion matrices and ablation tables for every model |
| `config.yaml` | Words, feature layout, split, training and segmenter settings |
| `NOTES.md` | Design reasoning, alternatives considered and interview questions, per milestone |

`NOTES.md` is the long version: why each choice was made and what was tried instead.
