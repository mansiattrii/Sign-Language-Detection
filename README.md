# Real-time Indian Sign Language word recognition

Recognizes isolated ISL words from a webcam using MediaPipe Holistic landmarks and an LSTM,
and builds a running sentence on screen. Work in progress; the full write-up comes at the end.

## Setup (Windows, PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest
```

## Data

The INCLUDE dataset (Zenodo record 4010759) is placed by hand under `data/raw/<word>/`, with the
folder names listed in `config.yaml`, e.g. `data/raw/hello/MVI_0029.MOV`.

## Feature extraction

```powershell
python -m src.extract --limit 2        # quick check: 2 videos per word
python -m src.extract                  # all videos -> data/processed/include.npz
python scripts/visualize_landmarks.py --video data/raw/hello/MVI_0029.MOV --out results/figures/landmarks_hello.png
```

## Split, baseline and evaluation

```powershell
python -m src.dataset                                   # writes data/processed/split.json (once)
python -m src.train --model rf                          # Random Forest baseline -> models/rf.joblib
python -m src.evaluate --model models/rf.joblib         # metrics, confusion matrix -> results/rf/
```

## LSTM, GRU and the feature ablation

```powershell
python -m src.train --model lstm                        # -> models/lstm.joblib + models/lstm.keras
python -m src.evaluate --model models/lstm.joblib       # -> results/lstm/
python -m src.ablation                                  # 3 feature sets x 3 seeds -> results/ablation.md
python -m src.ablation --model gru                      # same for the GRU -> results/ablation_gru.md
python -m src.ablation --model rf                       # same for the Random Forest -> results/ablation_rf.md
```

The ablation appends each run to its CSV as soon as it finishes, so an interrupted run resumes
when started again (`--fresh` starts over).

## Live app

```powershell
python -m src.app                                        # webcam; c clear, Backspace delete, q quit
python -m src.app --debug                                # also shows the motion value the segmenter uses
python -m src.app --practice --signer mansi --takes 5    # prompts each word, saves every take
python -m src.app --video data/raw/hello/MVI_0029.MOV --headless
python scripts/replay_segmenter.py                       # segmenter + model on the test split, no webcam
```

Practice mode saves clips to `data/own_recordings/<word>/<signer>_<n>.mp4` and a per-session log
to `data/own_recordings/practice_logs/`. In practice mode, r deletes the last take and n skips a word.

## Own recordings (new-signer test)

Record with practice mode (above) or with fixed-length takes:

```powershell
python scripts/record_signs.py --word hello --signer mansi --count 5   # space record, r redo, q quit
python -m src.extract --source own                                     # -> data/processed/own.npz
python -m src.evaluate --model models/lstm.joblib                      # -> results/lstm/own/
```

The own-recordings report also gives INCLUDE test accuracy on the same words, for a fair comparison
when only some words were recorded.
