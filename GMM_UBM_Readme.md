# GMM-UBM Speaker Verification: Current Pipeline and Why It Is Designed This Way

This document explains the UBM-GMM speaker verification system currently implemented in this repository, including what each stage does and why those choices were made.

The implementation described here is based on:

- `scripts/02_train_ubm.py`
- `scripts/03_enroll_targets.py`
- `scripts/04_run_short_eval.py`
- `src/features.py`
- `src/dsp_utils.py`
- `src/gmm_ubm.py`

## 1. What Model Are We Using?

The classical pipeline is:

1. Train a Universal Background Model (UBM) as a Gaussian Mixture Model (GMM) on pooled background speech.
2. Adapt speaker-specific models from the UBM using each target speaker's enrollment audio.
3. Score a test utterance using log-likelihood ratio (LLR):

$$
\text{LLR}(X) = \log p(X \mid \lambda_{\text{target}}) - \log p(X \mid \lambda_{\text{UBM}})
$$

where $X$ is the utterance feature sequence.

## 2. Why GMM-UBM for This Task?

GMM-UBM is still a strong baseline for speaker verification because it is:

- Data-efficient compared to large neural models.
- Interpretable (likelihood-based decisions).
- Naturally compatible with short utterance scoring through frame-level likelihood accumulation.
- Easy to deploy and debug in low-resource settings.

## 3. Feature Pipeline

Feature extraction is done with `extract_mfcc_features` in `src/features.py`.

### Implemented feature configuration

- Sample rate: 16 kHz
- MFCCs: 20
- STFT window (`n_fft`): 400
- Hop length: 160
- Deltas: first and second order
- Final dimension: 60 (`20 + 20 + 20`)

The feature matrix is normalized per utterance (z-normalization):

$$
\tilde{x}_{t,d} = \frac{x_{t,d} - \mu_d}{\sigma_d + 10^{-8}}
$$

### Why these features?

- MFCCs model speaker-relevant vocal tract characteristics.
- Delta and delta-delta add short-term dynamics (how speech changes over time).
- Per-utterance normalization reduces loudness/channel scaling variability.

## 4. UBM Training

UBM training is performed in `scripts/02_train_ubm.py`.

### Implemented training setup

- Input file list: `data/lists/ubm_train_list.txt`
- Preprocessing: silence trimming (`librosa.effects.trim`, `top_db=25`)
- Model: `GaussianMixture`
  - `n_components=64`
  - `covariance_type='diag'`
  - `max_iter=100`
  - EM training on all stacked feature frames
- Saved model: `results/models/ubm_model.pkl`

### Why this setup?

- `64` mixtures gives enough acoustic resolution while keeping compute practical.
- Diagonal covariance is more stable and less data-hungry than full covariance.
- Silence trimming focuses the UBM on speech-bearing regions and removes low-information frames.

## 5. Speaker Enrollment (Target Models)

Enrollment is handled by `scripts/03_enroll_targets.py`.

### Implemented enrollment flow

1. Load UBM from `results/models/ubm_model.pkl`.
2. Read `data/lists/test_enrollment_list.txt`.
3. For each speaker, use only files with `SX` in filename for enrollment.
4. Extract and stack 60-D features from enrollment utterances.
5. Clone UBM and adapt using short constrained EM updates:
   - `warm_start=True`
   - `max_iter=3`
6. Save one model per speaker at `results/enrolled_models/speaker_<id>.pkl`.

### Why adapt from UBM instead of training per-speaker from scratch?

- Enrollment data is limited; standalone per-speaker GMMs overfit easily.
- UBM acts as a strong prior and adaptation keeps models well-regularized.
- Short adaptation updates move only as much as enrollment evidence supports.

## 6. Short-Duration Evaluation Protocol

Evaluation is performed in `scripts/04_run_short_eval.py`.

### Implemented evaluation steps

1. Load UBM and all enrolled speaker models.
2. For each speaker, use `SI` files as test trials.
3. For each test file:
   - Extract longest word using `.WRD` timing via `extract_longest_word`.
   - Skip too-short words (`< 0.1 s`) to avoid unstable delta features.
   - Compute 60-D features.
4. Compute UBM score and target score (average log-likelihood per frame).
5. Form LLR score for:
   - Genuine trial: test speaker vs own model.
   - Impostor trials: same test segment vs 3 random other speaker models.
6. Compute ROC and Equal Error Rate (EER).
7. Plot score distributions.

### Why this evaluation design?

- LLR normalizes target likelihood by background likelihood, reducing common-acoustic bias.
- Using longest-word segments stresses short-utterance robustness.
- Genuine vs impostor distributions show separability directly.
- EER is a standard operating-point summary for verification systems.

## 7. About MAP Adaptation Utility in `src/gmm_ubm.py`

`src/gmm_ubm.py` includes a Reynolds-style mean-only MAP adaptation function (`map_adapt`) with relevance factor control.

This is conceptually the textbook adaptation path, but the current enrollment script (`03_enroll_targets.py`) uses warm-start EM fine-tuning instead.

### Why this matters

- Warm-start EM can update more than means and may fit enrollment data more aggressively.
- Mean-only MAP is often more conservative and robust with very little enrollment data.

If you need stricter classical behavior, wiring `map_adapt` into enrollment is the direct next step.

## 8. How To Run

Run from the `speaker_verification` root.

### 1) Prepare file lists

```bash
python scripts/01_prep_timit.py
```

### 2) Train UBM

```bash
python scripts/02_train_ubm.py
```

### 3) Enroll target speakers

```bash
python scripts/03_enroll_targets.py
```

### 4) Run short-duration evaluation

```bash
python scripts/04_run_short_eval.py
```

Optional comparison/analysis scripts:

```bash
python scripts/05_compare_systems.py
python scripts/06_duration_eval.py
```

## 9. Output Artifacts

- UBM model:
  - `results/models/ubm_model.pkl`
- Enrolled speaker models:
  - `results/enrolled_models/speaker_<id>.pkl`
- Evaluation and comparison plots:
  - `results/plots/` (multiple figures, including duration and DET comparisons)

## 10. Limitations and Practical Improvements

Current limitations:

- Enrollment adaptation currently uses warm-start EM, not explicit mean-only MAP in the script.
- Impostor sampling uses 3 random speakers per test segment (fast but stochastic).
- No explicit score calibration (for example, logistic calibration).

Practical improvements:

1. Switch enrollment to `map_adapt` with tuned relevance factor.
2. Add fixed random seed and repeated runs for stable confidence intervals.
3. Add score normalization/calibration for better threshold transfer.
4. Compare wake-word-only vs full-utterance GMM scoring in the same script.

Even with these limitations, this UBM-GMM pipeline is a valid and interpretable baseline for short-duration speaker verification, and it provides a strong reference point against the LSTM system.