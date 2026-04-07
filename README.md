# Speaker Verification (TIMIT)

A scaffolded speaker verification project using a classical **GMM-UBM** pipeline over the TIMIT corpus.

This README documents the **current setup exactly as implemented** in this repository, including:
- environment setup,
- sequential run order,
- expected outputs,
- and a complete file/module breakdown.

---

## 1) Current Implementation Status

This codebase is a **working scaffold** with mixed maturity:

- **Implemented and usable now**
  - TIMIT path scanning and list generation (`scripts/01_prep_timit.py`)
  - DSP helper for longest-word extraction from `.WRD` labels (`src/dsp_utils.py`)
  - MFCC + delta + delta-delta feature extraction (`src/features.py`)
  - Core GMM classes and wrappers (`src/gmm_ubm.py`)

- **Placeholder scripts (structure ready, full pipeline pending)**
  - UBM training driver (`scripts/02_train_ubm.py`)
  - Enrollment driver (`scripts/03_enroll_targets.py`)
  - Evaluation driver (`scripts/04_run_short_eval.py`)

This means you can run the full script sequence today, but steps 2–4 currently produce placeholder artifacts/logs.

---

## 2) Repository Layout

```text
speaker_verification/
├── README.md
├── requirements.txt
├── test_dsp.py
├── test_ubm.py
├── data/
│   ├── raw_timit/
│   │   ├── data/
│   │   │   ├── TRAIN/
│   │   │   └── TEST/
│   └── lists/
├── scripts/
│   ├── 01_prep_timit.py
│   ├── 02_train_ubm.py
│   ├── 03_enroll_targets.py
│   └── 04_run_short_eval.py
├── src/
│   ├── data_io.py
│   ├── dsp_utils.py
│   ├── features.py
│   └── gmm_ubm.py
└── results/
    ├── models/
    ├── enrolled_models/
    ├── scores/
    └── plots/
```

---

## 3) Prerequisites

- Python 3.9+ (project currently contains a local virtual env under `A1 Word Boundary/sp.venv`, but creating a project-local env is recommended)
- TIMIT dataset converted/extracted such that `.WAV.wav` and `.WRD` files exist under:
  - `data/raw_timit/data/TRAIN/...`
  - `data/raw_timit/data/TEST/...`

Install dependencies:

```bash
pip install -r requirements.txt
```

`requirements.txt`:
- `librosa`
- `scikit-learn`
- `soundfile`
- `numpy`
- `matplotlib`

---

## 4) Data Placement (Required)

Expected root:

```text
data/raw_timit/data/
```

Minimum expected structure:

```text
data/
  raw_timit/
    TIMITDIC.TXT
    SPKRINFO.TXT
    SPKRSENT.TXT
    data/
      TRAIN/
        DR1/.../<speaker>/*.WAV.wav
        ...
      TEST/
        DR1/.../<speaker>/*.WAV.wav
        ...
```

Notes:
- `scripts/01_prep_timit.py` scans for `*.WAV.wav` and skips utterances starting with `SA`.
- `src/dsp_utils.py` expects `.WRD` files next to corresponding `.WAV.wav` files.

---

## 5) Recommended Execution Order (Sequential)

Run all commands from the repository root: `speaker_verification/`.

### Step 1 — Generate Training/Enrollment Lists

```bash
python scripts/01_prep_timit.py
```

Produces:
- `data/lists/ubm_train_list.txt`
- `data/lists/test_enrollment_list.txt`

### Step 2 — Run UBM Training Driver (Current Placeholder)

Use explicit arguments to avoid outdated defaults in the script:

```bash
python scripts/02_train_ubm.py \
  --lists data/lists \
  --out results/models/ubm.pkl
```

Current behavior:
- creates output directory
- writes placeholder file: `results/models/ubm.placeholder.npy`

### Step 3 — Run Enrollment Driver (Current Placeholder)

```bash
python scripts/03_enroll_targets.py \
  --models results/models \
  --lists data/lists
```

Current behavior:
- ensures models directory exists
- prints enrollment placeholder status

### Step 4 — Run Short-Duration Evaluation Driver (Current Placeholder)

```bash
python scripts/04_run_short_eval.py \
  --models results/models \
  --scores results/scores \
  --plots results/plots
```

Current behavior:
- creates `results/scores/` and `results/plots/`
- prints evaluation placeholder status

---

## 6) Code Reading Order (For New Contributors)

If your goal is to understand the codebase quickly, follow this order:

1. `scripts/01_prep_timit.py`
   - Understands corpus traversal, file filtering, and list generation format.
2. `src/dsp_utils.py`
   - Understands how longest word segments are extracted from `.WRD` labels.
3. `src/features.py`
   - Understands MFCC feature pipeline and shape conventions.
4. `src/gmm_ubm.py`
   - Understands model objects (`UBMModel`, `TargetModel`) and adaptation scaffold.
5. `src/data_io.py`
   - Understands utility loaders/list readers and split helpers.
6. `scripts/02_train_ubm.py` → `03_enroll_targets.py` → `04_run_short_eval.py`
   - Understands experiment orchestration points to complete next.

---

## 7) File-by-File Breakdown

### Top-level

- `README.md`
  - Project runbook and structure documentation.
- `requirements.txt`
  - Python dependencies required by current code.
- `test_dsp.py`
  - Manual/visual DSP validation script for longest-word extraction and overlay plot.
- `test_ubm.py`
  - Manual inspection script for an already-trained UBM pickle and its mixture weights.

### `scripts/`

- `01_prep_timit.py`
  - Scans `TRAIN` for UBM candidates and `TEST` for speaker enrollment grouping.
  - Writes generated lists under `data/lists/`.
- `02_train_ubm.py`
  - Training entrypoint scaffold; currently writes placeholder output.
- `03_enroll_targets.py`
  - Enrollment entrypoint scaffold; currently ensures directories and prints status.
- `04_run_short_eval.py`
  - Evaluation entrypoint scaffold; currently creates result folders and prints status.

### `src/`

- `data_io.py`
  - Audio loading (`soundfile`), generic list reading, and split helper functions.
- `dsp_utils.py`
  - Word-level segment extraction from transcript timing (`.WRD`) for a wav file.
- `features.py`
  - Feature engineering: MFCC + first and second deltas.
- `gmm_ubm.py`
  - GMM wrappers, save/load helpers, UBM training function, and MAP-adaptation scaffold.

### `data/` and `results/`

- `data/raw_timit/`
  - Original corpus metadata + TRAIN/TEST tree.
- `data/lists/`
  - Generated path/list manifests consumed by scripts.
- `results/`
  - Experiment artifacts (models, scores, plots, enrolled models).

---

## 8) Optional Validation Scripts

### DSP extraction sanity check

```bash
python test_dsp.py
```

### UBM inspection (requires an actual trained pickle at `results/models/ubm_model.pkl`)

```bash
python test_ubm.py
```

---

## 9) Known Caveats in Current Setup

- Scripts `02_*`, `03_*`, `04_*` are placeholders and do not yet perform full speaker verification.
- Use explicit CLI arguments for scripts `02_*` to `04_*`; their internal defaults still reference an older folder name (`short_duration_sv/...`).
- `test_dsp.py` and `test_ubm.py` are analysis/inspection scripts, not automated unit tests.

---

## 10) Next Development Targets (Suggested)

1. Replace placeholder logic in `02_train_ubm.py` by:
   - loading files from `ubm_train_list.txt`,
   - extracting frame-level features,
   - fitting and saving a real UBM.
2. Implement true MAP adaptation in `03_enroll_targets.py`.
3. Implement genuine likelihood-ratio scoring and EER/DET reporting in `04_run_short_eval.py`.

This will transition the repository from scaffold mode into a full reproducible SV experiment pipeline.