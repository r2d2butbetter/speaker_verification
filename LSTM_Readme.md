# LSTM Speaker Verification: Current Model and Why It Is Designed This Way

This README describes the LSTM system that is currently used in this repository, based on the actual implementation in `lstm/model.py`, `lstm/data.py`, `lstm/train.py`, and `lstm/enroll_and_eval.py`.

The goal is to learn a speaker embedding (d-vector) that stays close for the same speaker and far apart for different speakers, especially for short utterances.

## 1. What Model Are We Using?

We use `SpeakerLSTM`, a bidirectional LSTM encoder with attention pooling and metric-learning training.

### Architecture (as implemented)

- Input features: 39 dims per frame (13 MFCC + delta + delta-delta)
- Encoder: 2-layer BiLSTM, hidden size 256 per direction
- Sequence output size: 512 per frame (forward + backward)
- Pooling: learned self-attention over time
- Projection head: Linear(512 -> 128) + ReLU
- Output embedding: L2-normalized 128-d vector
- Classifier head: optional in class definition, not used in current training (`num_classes=None`)

## 2. Why This Architecture?

### Why BiLSTM?

Speech is sequential, and speaker traits are spread across time. A BiLSTM captures both past and future context at each frame, which helps when phonetic content is short or ambiguous.

### Why attention pooling?

Not all frames are equally useful for speaker identity. Attention gives higher weight to informative regions and lower weight to silence/noisy frames, instead of averaging everything equally.

### Why 128-d normalized embeddings?

L2 normalization places embeddings on a unit hypersphere, making cosine-based scoring stable and consistent. This is a standard setup for verification systems.

## 3. Feature and Data Pipeline

### Feature extraction (`src/features.py` + `lstm/data.py`)

- Audio is loaded at 16 kHz.
- MFCC config for this pipeline: 13 coefficients, `n_fft=400`, `hop_length=160`.
- Deltas and delta-deltas are stacked to produce 39-dim features.
- Feature tensor shape fed to LSTM: `(time, 39)`.

### Training dataset behavior (`ShortUtteranceDataset`)

- Input list: `data/lists/ubm_train_list.txt`.
- Triplets are generated online:
	- Anchor: one utterance from speaker A
	- Positive: another utterance from speaker A (or same if only one exists)
	- Negative: utterance from a different speaker
- Each utterance is random-cropped to 1.5 s during training; shorter utterances are zero-padded.

### Evaluation dataset behavior

- In eval mode (`is_train=False`), full utterance is used (no crop/pad in dataset).
- This allows scoring with complete speech context.

## 4. Training Setup

Current training settings from `lstm/train.py`:

- Loss: `TripletMarginLoss(margin=1.0, p=2)`
- Optimizer: Adam, learning rate `1e-3`, weight decay `1e-5`
- LR scheduler: `StepLR(step_size=4, gamma=0.5)`
- Epochs: 25
- Batch size: 32
- Gradient clipping: `max_norm=5.0`

Why each setting is used:

- `margin=1.0` in triplet loss sets a clear separation target between positive and negative pairs without making optimization too brittle.
- `p=2` uses Euclidean distance, matching common metric-learning practice for embedding spaces.
- Adam with `lr=1e-3` gives stable early convergence for recurrent networks.
- `weight_decay=1e-5` adds light regularization to reduce overfitting.
- `StepLR(step_size=4, gamma=0.5)` decays learning rate every 4 epochs so training starts fast, then fine-tunes with smaller updates.
- `epochs=25` provides enough passes for convergence in this setup while keeping training time practical.
- `batch_size=32` balances gradient stability and memory use.
- `max_norm=5.0` prevents exploding gradients in LSTM backpropagation.

Per-utterance normalization is applied in the training loop before model forward pass:

`x_norm = (x - mean_time) / (std_time + 1e-6)`

## 5. Why This Training Strategy?

### Why triplet loss instead of speaker classification loss?

Verification is a similarity problem, not closed-set classification. Triplet loss directly optimizes relative distances:

- pull Anchor and Positive together
- push Anchor and Negative apart by at least a margin

This produces embeddings that generalize better to unseen speakers.

### Why 1.5-second random chunks during training?

Short chunks regularize the model for short-utterance verification and increase sample diversity through random segment selection.

### Why not 0.5-second chunks?

0.5 seconds is usually too short to provide enough stable speaker evidence (especially across varying phonetic content). 1.5 seconds is a practical compromise: short enough to train for short-utterance use cases, but long enough to retain consistent speaker traits.

### Why gradient clipping?

RNN-based models can produce unstable gradients. Clipping prevents exploding gradients and improves convergence stability.

## 6. Enrollment and Evaluation Flow

Evaluation in `lstm/enroll_and_eval.py` works as follows:

1. Read `data/lists/test_enrollment_list.txt`.
2. For each speaker entry, use first 3 files for enrollment and remaining files for testing.
3. Compute embeddings for enrollment and test utterances.
4. Build one target model per speaker by averaging enrollment embeddings and re-normalizing.
5. Score every test embedding against every enrolled speaker using cosine similarity (dot product on normalized vectors).
6. Compute ROC and report EER.

## 7. Why This Evaluation Design?

### Why average enrollment embeddings?

Averaging multiple enrollment utterances reduces utterance-level noise and produces a more robust speaker template.

### Why cosine scoring?

With L2-normalized embeddings, cosine similarity is simple, fast, and effective for verification.

### Why EER as primary metric?

EER is threshold-independent at the operating point where false accept rate equals false reject rate, making it a standard summary metric for speaker verification.

## 8. How To Run

Run commands from `speaker_verification` root.

### 1) Prepare file lists

```bash
python scripts/01_prep_timit.py
```

### 2) Train LSTM

```bash
python lstm/train.py
```

### 3) Evaluate default final checkpoint

```bash
python lstm/enroll_and_eval.py
```

### 4) Evaluate a specific checkpoint

```bash
python lstm/enroll_and_eval.py lstm_epoch_25.pt
```

## 9. Output Artifacts

- Epoch checkpoints: `results/models/lstm/lstm_epoch_<n>.pt`
	- Saved as training checkpoint dict (model, optimizer, epoch, loss)
- Final model: `results/models/lstm/lstm_final.pt`
	- Saved as model `state_dict`

Evaluation prints trial count and EER to console.

## 10. Current Limitations and Next Improvements

- No hard/semi-hard negative mining yet.
- No augmentation (noise/reverb/speed perturbation) yet.
- No explicit VAD; attention is expected to down-weight less useful regions.
- A dedicated validation protocol per epoch would make model selection more reliable.

Even with these limits, the current design is a strong baseline for short-utterance speaker verification because it combines temporal modeling (BiLSTM), frame importance learning (attention), and verification-oriented metric learning (triplet loss).

## 11. Clarification on the "0.5 rate"

The `0.5` value in training is the learning-rate decay factor (`gamma=0.5`) in `StepLR`, not a 0.5-second audio duration.

- Duration used for training chunks: `chunk_length_sec=1.5`.
- Meaning of `gamma=0.5`: every scheduler step, LR becomes half of the previous value.
- Example with `step_size=4`: after epoch 4, LR goes from `1e-3` to `5e-4`; after epoch 8, to `2.5e-4`, and so on.

So to be explicit: we are **not** training the LSTM on 0.5-second speech segments. We train on 1.5-second random chunks, and only the optimizer learning rate is multiplied by 0.5 at scheduler steps.

## 12. Why Only DET, EER Bar, and ROC Are Kept in Comparison Plots

In `scripts/05_compare_systems.py`, we keep only three summary plots:

- **DET curve**: best visualization of verification trade-off (FAR vs FRR) at all thresholds.
- **EER bar chart**: fastest single-number comparison between systems.
- **ROC curve**: complementary ranking-quality view (TPR vs FPR).

Why the others were dropped:

- Score histograms, threshold-sweep curves, and box plots are useful for deep debugging, but they are redundant for the final model comparison story.
- Keeping only DET + EER + ROC makes the report cleaner while still showing that LSTM outperforms GMM-UBM from multiple angles.
- Fewer plots reduce visual clutter and make the main conclusion easier to communicate.
