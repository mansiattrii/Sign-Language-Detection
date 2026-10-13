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
- **No signer IDs.** File names are camera counters. They come in runs of 3-4 consecutive numbers
  per word, and neighbouring words share a sitting. Milestone 2 groups whole sittings (see below).
  Own recordings are the real signer-independent test.
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


## Milestone 2: split, augmentation and Random Forest baseline

### What was built

- **`dataset.py`.** Groups clips into recording *sittings* and saves a fixed split to
  `data/processed/split.json` (indices plus file paths, so a changed dataset is detected and
  refused). Every model and every ablation run loads this same file. Creating a new split needs
  `--force`.
- **`augment.py`.** Scaling, small 2D rotation, Gaussian noise on detected points only, time
  warping with boundary jitter, and an optional horizontal flip (off by default). Unit-tested,
  including the double flip returning the original clip exactly.
- **`models.py`, `train.py`, `evaluate.py`.** Random Forest on per-clip mean and standard deviation
  of every feature. `evaluate.py` writes top-1, top-5, macro F1, per-class accuracy, a confusion
  matrix PNG and the five most-confused pairs, and evaluates own recordings separately once they
  exist.
- Resampling to 30 frames was already built in Milestone 1 (`features.resample`); Milestone 2
  reuses its interpolation for time warping.

### The split, and the leak it took two attempts to remove

1. **First attempt: group by session within each word.** Runs of consecutive camera numbers per
   word were kept together. The Random Forest scored 50.0% top-1, but the confusion matrix showed
   pairs swapped in *both directions, every time*: all GOOD test clips predicted as BAD and all BAD
   as GOOD; the same for I/YOU and GOOD MORNING/GOOD NIGHT.
2. **Diagnosis.** GOOD is 5158-5160 and BAD is 5161-5163: one person signed both in one sitting.
   If that sitting's GOOD clips are in test but its BAD clips are in train, the nearest thing the
   model has seen is the same person signing BAD. Real confusion between similar signs is partial;
   a clean swap points to a leak.
3. **Fix: group by sitting across words.** Clips whose camera numbers are within 10 of each other,
   across all words, form a sitting. Greetings and pronouns were filmed on cameras with overlapping
   counters, which chained them into blocks of 110+ clips; blocks over 60 clips are re-split at
   gaps above 5. Thresholds err towards bigger groups, because merging two sittings is harmless and
   splitting one leaks. Result: 63 sittings.
4. **Assigning sittings.** A sitting spans many words, so the split can't be stratified per word.
   A seeded search tries 2000 random orders, fills test then val greedily towards about 20% of each
   word, rejects any split that leaves a word missing from train, val or test, and keeps the
   closest one. Final split: train 382 / val 102 / test 136 clips; every word has 3-7 test clips.
5. **What the test set means.** It is *sitting-independent*, not signer-independent: the same
   person can return on another day. Own recordings (Milestone 4) are the signer-independent test.

### Augmentation choices

- **Uniform speed changes do nothing here.** Every clip is trimmed and resampled to 30 frames, so
  playing it 20% faster and resampling gives the same 30 frames. Time warping therefore changes
  the speed *within* the clip (`u ** gamma`: faster at the start and slower at the end, or the
  reverse) and crops up to 10% at each end, which also imitates the live segmenter cutting a sign
  slightly early or late.
- **Rotation and scaling act about each block's origin** (wrist, shoulder midpoint), which is
  where the normalized coordinates are already centred, so no extra centring is needed.
- **Noise only on detected points.** A missing hand must stay exactly zero; otherwise noise would
  make "no hand" look like a tiny hand.
- **Flip is off by default.** Mirroring swaps the dominant hand. That is a valid left-handed
  signer, but for signs that are not symmetric it can teach the wrong thing, so it is an option to
  test, not a default.
- **Flip implementation:** one column permutation (left hand <-> right hand, flags swapped, pose
  pairs 11/12, 13/14, 15/16, face mirror pairs) plus negating every x. The permutation is its own
  inverse, which is why the double-flip test passes exactly.

### Baseline results (INCLUDE test split, 136 clips, seed 42)

| Model | Top-1 | Top-5 | Macro F1 |
|---|---|---|---|
| Random Forest, mean + std, all features | 0.581 | 0.926 | 0.530 |
| Same, plus 5 augmented copies per clip | 0.596 | 0.912 | 0.559 |

Validation top-1: 0.569 without augmentation, 0.598 with. The augmentation difference is 2 test
clips on one seed, so it is not a meaningful effect. Most-confused pairs (no augmentation): GOOD
MORNING/GOOD NIGHT (6 errors), MARKET->FATHER (4), GOOD MORNING->GOOD (4), SISTER->BROTHER (3),
FATHER->SICK (3).

### A hypothesis that turned out wrong

Some errors were still total: all HOT clips predicted as SCHOOL, all MARKET clips as FATHER. My
guess was that the forest recognised the *person* from body proportions. A forest trained on
clip-averaged face shape alone scored 2.2% on test (chance is 3.3%), so static body shape does not
identify words across sittings. The odd pairs also remain when the forest sees hands only. The
likely cause is that the test sitting's signer uses a different variant of those signs, which is
a property of the data and a limitation to state, not something the pipeline can fix.

Side note from the same diagnostic: hands + pose scored 0.669 vs 0.581 for all features with the
forest. One seed with a Random Forest; the real comparison is the LSTM ablation over 3 seeds.

### Alternatives considered

- *Random split:* takes of the same word from one sitting land on both sides; accuracy would look
  far better than it is.
- *scikit-learn `StratifiedGroupKFold`:* tried first; it can leave a word out of a fold entirely,
  and with sittings spanning words it cannot stratify exactly anyway.
- *Leave-one-signer-out:* the right protocol, but INCLUDE has no signer labels.

### Likely interview questions

- **How did you split data without signer IDs?** Grouped clips into recording sittings using camera
  file numbers across words, and put whole sittings in one split. I found the need for
  cross-word grouping from a confusion matrix with perfectly swapped pairs.
- **Why is per-word grouping not enough?** One person records many words in a row. If their GOOD is
  in test and their BAD in train, the model can match the person rather than the sign.
- **Why does time warping not just change the speed?** Clips are resampled to a fixed length, so a
  uniform speed change cancels out. Warping varies speed within the clip and jitters the boundaries.
- **Why mean and standard deviation for the baseline?** They are the simplest fixed-size summary of a
  sequence. They ignore order, which is exactly what the LSTM comparison tests.
- **Why is top-5 so much higher than top-1?** Many errors are between close pairs (GOOD MORNING/GOOD
  NIGHT, BROTHER/SISTER): the right word is usually among the first few guesses.
