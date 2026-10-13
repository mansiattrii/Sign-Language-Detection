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
