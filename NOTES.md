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

## Milestone 3: LSTM, feature ablation and GRU comparison

### What was built

- `build_recurrent` in `src/models.py`: LSTM(128, return_sequences) -> Dropout(0.3) -> LSTM(64) ->
  Dropout(0.3) -> Dense(64, relu) -> Dense(30, softmax), Adam, sparse categorical cross-entropy.
  It takes the input feature size as a parameter, so the ablation runs the identical architecture
  on 128, 149 or 308 features. `cell="gru"` swaps in GRU layers of the same sizes.
- Training (`src/train.py`): a `PyDataset` that reshuffles and freshly augments the training clips
  every epoch, so the model never sees the same augmented clip twice. Early stopping restores the
  weights of the best epoch. Class weights switch on only if the largest class has 1.5x the clips
  of the smallest; with 20-21 clips per word they stay off. Seeds go through
  `keras.utils.set_random_seed` plus TensorFlow op determinism: retraining the final model
  reproduced its ablation run exactly (val 0.824, test 0.706).
- `src/ablation.py`: 3 feature configurations x 3 seeds (42, 43, 44), each evaluated on the same
  fixed test split. Writes every run to the CSV as it finishes (resumable) and a Markdown summary
  with mean ± sample standard deviation.

### Two training decisions, both made on validation only

- **Early stopping watches validation accuracy, not loss.** Without early stopping, validation loss
  bottomed out around epoch 25 while validation accuracy kept rising until about epoch 120. The model
  becomes over-confident on the clips it gets wrong, which raises the loss even as more clips are
  classified correctly. Stopping on loss stopped at epoch 26 with val top-1 0.598. Patience is 50
  epochs because accuracy on 102 clips jumps by several points from one epoch to the next.
- **Augmentation stays on.** All features, 3 seeds: val top-1 0.686 ± 0.017 with augmentation,
  0.673 ± 0.057 without. Same mean within noise; augmentation made seeds more consistent. These are
  best-epoch validation scores, so both are optimistic.

### Ablation results (INCLUDE test split, 136 clips, mean ± std over 3 seeds)

| Model | Features | Val top-1 | Test top-1 | Test top-5 | Test macro F1 |
|---|---|---|---|---|---|
| LSTM | hands (128) | 0.817 ± 0.030 | 0.752 ± 0.021 | 0.985 ± 0.013 | 0.760 ± 0.015 |
| LSTM | hands + pose (149) | 0.863 ± 0.052 | 0.757 ± 0.060 | 0.968 ± 0.015 | 0.751 ± 0.060 |
| LSTM | hands + pose + face (308) | 0.686 ± 0.017 | 0.691 ± 0.072 | 0.922 ± 0.008 | 0.687 ± 0.093 |
| GRU | hands (128) | 0.791 ± 0.091 | 0.694 ± 0.011 | 0.949 ± 0.013 | 0.697 ± 0.017 |
| GRU | hands + pose (149) | 0.840 ± 0.023 | 0.679 ± 0.030 | 0.968 ± 0.011 | 0.677 ± 0.048 |
| GRU | hands + pose + face (308) | 0.673 ± 0.025 | 0.676 ± 0.000 | 0.931 ± 0.011 | 0.674 ± 0.028 |
| Random Forest | hands (128) | 0.569 ± 0.026 | 0.591 ± 0.008 | 0.907 ± 0.004 | 0.552 ± 0.006 |
| Random Forest | hands + pose (149) | 0.667 ± 0.026 | 0.681 ± 0.011 | 0.941 ± 0.007 | 0.630 ± 0.013 |
| Random Forest | hands + pose + face (308) | 0.601 ± 0.028 | 0.593 ± 0.011 | 0.934 ± 0.007 | 0.547 ± 0.017 |

Every configuration includes FLAGS (the 2 hand-present columns). Per-run tables are in
`results/ablation.md`, `results/ablation_gru.md` and `results/ablation_rf.md`.

### What the ablation showed

- **Temporal order matters.** The LSTM beats the Random Forest on mean/std features in every
  configuration (hands: 0.752 vs 0.591). Pairs that differ mainly in order or movement path improved
  most: the forest confused GOOD MORNING and GOOD NIGHT 6 times, the final LSTM once.
- **The face hurts.** Adding 159 face columns lowers LSTM test top-1 from about 0.75 to 0.691 and
  makes it less stable across seeds. With 382 training clips, more than doubling the input size
  gives the network more ways to fit the training sittings. Most of the face subset describes who
  is signing (face shape) rather than what is signed, and the non-manual cues that do matter
  (eyebrows in questions) belong to sentence-level grammar, which is out of scope here.
- **Hands vs hands + pose is a tie on test** (0.752 ± 0.021 vs 0.757 ± 0.060). Pose wins on
  validation (0.863 vs 0.817). The Random Forest gains clearly from pose (0.591 -> 0.681): without
  temporal order, where the hands are relative to the body carries much of the information.
- **The GRU is consistently below the LSTM** on test (0.68-0.69 vs about 0.75), despite similar
  validation scores. The gap is about as large as the seed spread, so the claim is only that the
  GRU was worse here, not that GRUs are worse in general.
- **Validation is optimistic.** It picks the stopping epoch and has only 102 clips from a handful of
  sittings, so val-test gaps of 0.05-0.10 are expected. Test numbers are the ones to quote.

### Final model

`feature_groups` is now `[HANDS, POSE, FLAGS]`, chosen by validation accuracy (not test). The saved
`models/lstm.keras` is that configuration with the default seed 42, which happens to be the weakest
of its three seeds on test:

| Split | Top-1 | Top-5 | Macro F1 |
|---|---|---|---|
| Validation (102 clips) | 0.824 | 0.971 | 0.817 |
| Test (136 clips) | 0.706 | 0.956 | 0.703 |

The number to quote is the ablation mean, 0.757 ± 0.060, not a single seed. Most confused test
pairs: GOOD MORNING->GOOD (4), WE->MARKET (4), SISTER->BROTHER (3), DOG->HOW ARE YOU (3),
I->YOU (3). GOOD MORNING starts with the GOOD sign, so a clip that stresses its first half reads as
GOOD. HOT->SCHOOL (2 of 3) persists from the forest, consistent with the sign-variant explanation
in Milestone 2.

`results/rf/` from Milestone 2 was trained on all four groups; rerunning `src.train --model rf` now
uses hands + pose, the new default.

### Alternatives considered

- *Saving the best seed:* seed 43 scored 0.922 on validation. Choosing on validation would be
  legitimate, but one fixed seed keeps the pipeline simple and avoids presenting a lucky run as the
  result.
- *Masking layer for missing hands:* a missing hand is zeros plus a flag of 0, so the network
  already knows when a hand is absent, and every sequence has the same length, so there is no
  padding to mask.
- *Bidirectional LSTM, attention, 1D CNN, Transformer:* possibly small gains, but with 382 training
  clips the spread between seeds already exceeds the differences between configurations. More data
  would help more than a bigger model.
- *Monitoring validation loss with a learning-rate schedule:* would also avoid the early stop, but
  adds hyperparameters to tune on a 102-clip validation set.

### Likely interview questions

- **Why an LSTM and not averaged features?** Signs are movements: GOOD MORNING and GOOD NIGHT
  differ in the order and direction of the motion. The mean/std baseline cannot see order; the LSTM
  gains about 16 points over it on hands-only features.
- **Why did adding the face make it worse?** Too many inputs for the amount of data (overfitting),
  and the face mostly encodes identity, not the isolated word. Feature choice is a modelling
  decision to test, not assume.
- **Why report mean ± std over seeds?** With 136 test clips one clip is 0.7 points, and seeds of
  the same configuration differ by up to 14 points. A single run cannot separate two configurations.
- **Why does validation loss rise while accuracy rises?** Cross-entropy punishes confident mistakes
  heavily. As the network fits the training sittings it becomes very confident, so the clips it
  still gets wrong cost a lot of loss even while more clips are classified correctly.
- **Is your test score the real-world accuracy?** No. It is accuracy on unseen recording sittings of
  INCLUDE signers in INCLUDE conditions. Own recordings (Milestone 4) test a new signer, camera and
  room.
