# Interview notes

## Milestone 1: setup, data and feature extraction

### What was built

- **Environment.** Python 3.11 with `mediapipe==0.10.21`. Releases from 0.10.30 onwards removed the
  legacy `mp.solutions.holistic` API, so the version is pinned. TensorFlow 2.17.1 needs `numpy<2`,
  which mediapipe 0.10.21 also wants, so the stack is consistent.
- **Vocabulary.** 30 words from INCLUDE, each with 20-21 videos, so the classes are balanced.
  13 are INCLUDE-50 words (hello, good morning, thank you, I, father, brother, good, happy, hot,
  new, house, car, dog); the other 17 come from the full set to make more natural sentences, so
  results are not directly comparable with INCLUDE-50 papers. INCLUDE has no verbs at all, so
  sentences are telegraphic ("I SICK HOSPITAL", "MOTHER HAPPY").
- **`landmarks.py`.** Wraps Holistic. Returns per-frame landmarks and stacks a clip into arrays with
  NaN for frames where a part was not detected. x and z are multiplied by width/height so all axes
  are in the same unit.
- **`features.py`.** The only place landmarks become model input. `build_sequence()` normalizes each
  frame, trims frames without hands at both ends, and resamples to 30 frames. Training and the live
  app both call it. `FeatureLayout` exposes the column slices (`HANDS`, `POSE`, `FACE`, `FLAGS`) so
  the ablation selects groups by name.
- **`extract.py`.** Two stages: (1) run Holistic once per video and cache raw landmarks (slow,
  resumable, `--limit N` per word); (2) build `data/processed/<source>.npz` from the cache (fast).

### Data facts found while inspecting INCLUDE

- Layout on Zenodo: `<Category>/<N>. <Word>/MVI_xxxx.MOV`, some `.MP4`, a few extra takes in
  `Extra/` subfolders. Downloaded into `data/raw/<word>/` so the loader only needs the word.
- **No signer IDs.** File names are camera counters. They come in runs of 3-4 consecutive numbers,
  which look like one recording session each. The split (Milestone 2) groups by session so takes
  from the same sitting never end up in both train and test. Own recordings are the real
  signer-independent test.
- **Face indices.** All 53 are valid MediaPipe face-mesh points. Two corrections: (1) MediaPipe names
  sides from the signer's view, so the "left" groups in the original spec are MediaPipe's *right*
  groups; (2) the other eye's list was reordered to `[263, 387, 385, 362, 380, 373]` so that point i
  mirrors point i, which flip augmentation needs. Verified by reflecting MediaPipe's canonical face
  model across its midline: every pair matches at distance 0.

### Why it was done this way

- **Landmarks instead of a CNN on raw frames.** About 600 videos is far too little to train a video
  CNN from scratch. Holistic was trained on large datasets and already removes background,
  clothing, lighting and skin tone. The model sees about 300 numbers per frame instead of millions
  of pixels, trains on a CPU in minutes, and runs live on a laptop.
- **Wrist-relative hand normalization.** Subtracting the wrist removes where the hand is in the
  image; dividing by wrist-to-middle-MCP removes how far the signer is from the camera. What is left
  is hand shape and orientation. Rotation is not removed, because orientation carries meaning in
  sign language.
- **Hand size measured in 3D, not 2D (found during the `--limit 2` run).** With the 2D
  wrist-to-MCP length, normalized fingertips reached 30 units on MARKET and 12 on GOOD NIGHT,
  where 2-3 is normal. Cause: foreshortening. When the hand points at the camera its 2D length
  collapses (to 4% of the clip median in the worst frame), so dividing by it explodes. The 3D
  length never dropped below 34% of the median, and fingertips stay within 3.3 units. Shoulders
  still use 2D width: they face the camera and the pose model's z is much noisier.
- **Shoulder-relative pose and face.** Same idea for the body: shoulder midpoint as origin, shoulder
  width as unit. This keeps *where* the hands and face are relative to the body, which is what
  distinguishes many signs (I vs YOU vs HE is mostly pointing direction).
- **Consequence for the ablation.** Because each hand is normalized to its own wrist, the hand
  blocks contain hand *shape* only. Hand *location* and *movement path* live in the pose block
  (wrists 15/16, elbows). So "hands only" sees shape over time, and adding pose adds location.
- **Missing hands as zeros plus a flag.** The network needs a fixed-size input. Zeros alone would be
  ambiguous with a real hand at the origin, so a presence flag tells the model the block is empty.
  Flags are not normalized; they are 0/1.
- **Missing shoulders.** Carry the last valid reference forward, back-fill frames before the first
  valid one (shoulders barely move during a sign), and fall back to the nose only if the clip never
  shows usable shoulders. Extraction logs how often that happens.
- **Aspect-ratio correction.** MediaPipe gives x as a fraction of width and y as a fraction of
  height. On a 16:9 video one unit of x is 1.78 times longer than one unit of y, so distances and
  angles would differ between INCLUDE (16:9) and a 4:3 webcam. Multiplying x by width/height fixes it.
- **Fixed length of 30 frames.** Clips vary in length (signing speed, frame rate). A fixed length
  lets clips be batched, lets the Random Forest use the same input, and makes the live app's input
  identical to training. Linear interpolation keeps the shape of the motion and only changes its
  speed.
- **Two-stage extraction with a raw cache.** Holistic is the slow part. Caching raw landmarks means
  changing normalization, the face subset or the sequence length only reruns the fast stage.

### Alternatives considered

- *Downloading the whole dataset (55 GB):* not needed, and more than the free disk. The zips on
  Zenodo were read remotely with HTTP range requests and only the 620 chosen videos were extracted.
- *Pad/truncate instead of resample:* padding wastes model capacity on zeros and truncation can cut
  off the end of a sign. Resampling keeps the whole sign.
- *Interpolating missing hands from neighbouring frames:* more realistic, but it invents data, and
  the live app would need the same rule. Zeros plus a flag is simpler and honest. A side effect: at
  the edge of a gap, resampling blends a real hand with zeros for one frame.
- *Hands solution plus Pose solution separately:* Holistic gives all parts in one consistent pass
  and labels hands by the body they belong to, which is more reliable than per-hand handedness.

### Likely interview questions

- **Why landmarks and not raw video?** Too little data for a video CNN. Landmarks remove nuisance
  factors (background, lighting, skin tone), are small, and run in real time on a CPU.
- **Why normalize relative to the wrist and the shoulders?** To remove position in the frame and
  distance from the camera, so the model learns the sign rather than where the person stood.
- **How are missing hands handled?** That hand's block is zero and its presence flag is 0. Leading
  and trailing frames without any hand are trimmed off.
- **Why resample to a fixed length?** Batching, the same input for every model, and the same input
  at inference time. Interpolation preserves the trajectory and only changes the tempo.
- **Your dataset has no signer IDs. How do you avoid leakage?** Group the split by recording session
  (consecutive camera file numbers), and test on new signers using my own recordings.
- **Why is MediaPipe's "left eye" on the right of the picture?** MediaPipe names sides from the
  signer's point of view. The same applies to hands, which is why frames must not be mirrored before
  Holistic, only for display.

### Extraction results (full run)

- 620 videos (20-21 per word), all 25 fps, 36-154 frames long (median 59).
- 620 clips kept, 0 skipped for too few hand frames, 0 failed videos, 0 clips needing the nose
  fallback. Output: `data/processed/include.npz`, `X` of shape (620, 30, 308).
- Holistic ran at 2.4-2.8 s per video on CPU (model_complexity 1, frames downscaled to 960 px).
  The raw landmark cache is 226 MB; the final dataset is 19 MB.
- After resampling, the left hand is present in 85% of frames and the right hand in 75%.
- Value check: hands within about +-3.4 (units of hand length). Pose and face x/y mostly within
  +-2 shoulder widths. The extremes (up to 6.5) are in pose z, MediaPipe's rough depth estimate,
  which is noisy. One clip (`cold/MVI_9333`) has a face 2.5 shoulder widths above the shoulders,
  probably from a few frames with misdetected shoulders. Left as is for now.
