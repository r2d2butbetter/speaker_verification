"""GMM-UBM short-duration evaluation: scoring, EER, and plots."""

import sys
from pathlib import Path
import argparse
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

sys.path.append(str(Path(__file__).parent.parent))

from src.data_io import load_wav
from src.features import mfcc_with_deltas
from src.gmm_ubm import UBMModel, TargetModel


def extract_feats(wav_path: str) -> np.ndarray:
    """Load wav and return feature matrix (T, 39)."""
    audio, sr = load_wav(wav_path)
    feats = mfcc_with_deltas(np.array(audio), sr)
    return feats.T.astype(np.float64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ubm", type=Path, default=Path("results/models/ubm_model.pkl"))
    ap.add_argument("--enrolled", type=Path, default=Path("results/enrolled_models"))
    ap.add_argument("--lists", type=Path, default=Path("data/lists"))
    ap.add_argument("--scores-dir", type=Path, default=Path("results/scores"))
    ap.add_argument("--plots-dir", type=Path, default=Path("results/plots"))
    ap.add_argument("--n-enroll", type=int, default=3,
                    help="Must match enrollment split used in 03_enroll_targets.py")
    args = ap.parse_args()

    for p in [args.scores_dir, args.plots_dir]:
        p.mkdir(parents=True, exist_ok=True)

    # ---- Load UBM ----
    if not args.ubm.exists():
        print(f"UBM not found at {args.ubm}. Train it first.")
        return
    ubm = UBMModel.load(args.ubm)
    print(f"Loaded UBM ({ubm.gmm.n_components} components).")

    # ---- Load enrolled target models ----
    target_models = {}
    for pkl in sorted(args.enrolled.glob("speaker_*.pkl")):
        spk_id = pkl.stem.replace("speaker_", "")
        target_models[spk_id] = joblib.load(pkl)
    if not target_models:
        print("No enrolled models found. Run scripts/03_enroll_targets.py first.")
        return
    print(f"Loaded {len(target_models)} enrolled speaker models.")

    # ---- Build test trials from enrollment list ----
    enroll_list = args.lists / "test_enrollment_list.txt"
    if not enroll_list.exists():
        print("Missing test_enrollment_list.txt.")
        return

    test_trials = []  # (speaker_id, wav_path)
    with open(enroll_list) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) != 2:
                continue
            speaker_id = Path(parts[0]).name
            wav_paths = parts[1].split(",")
            # Test utterances = everything after the enrollment set
            for wp in wav_paths[args.n_enroll:]:
                test_trials.append((speaker_id, wp))

    if not test_trials:
        print("No test trials. Check enrollment list and --n-enroll.")
        return
    print(f"Scoring {len(test_trials)} test utterances against {len(target_models)} speakers...")

    # ---- Score ----
    y_true = []
    y_scores = []
    scored = 0

    for test_spk, wav_path in test_trials:
        try:
            X = extract_feats(wav_path)
        except Exception as e:
            print(f"  Skipping {wav_path}: {e}")
            continue

        ubm_score = ubm.gmm.score(X)

        for enr_spk, tgt_model in target_models.items():
            # Log-likelihood ratio
            tgt_score = tgt_model.gmm.score(X)
            llr = tgt_score - ubm_score
            y_scores.append(llr)
            y_true.append(1 if test_spk == enr_spk else 0)

        scored += 1
        if scored % 50 == 0:
            print(f"  Scored {scored}/{len(test_trials)} utterances")

    y_true = np.array(y_true)
    y_scores = np.array(y_scores)

    # ---- Save raw scores ----
    np.savez(args.scores_dir / "gmm_ubm_scores.npz", y_true=y_true, y_scores=y_scores)

    # ---- Compute EER ----
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fnr - fpr))
    eer = fpr[eer_idx]

    print(f"\n======================================")
    print(f"GMM-UBM Evaluation")
    print(f"Trials:              {len(y_scores)}")
    print(f"Equal Error Rate:    {eer * 100:.2f}%")
    print(f"======================================")

    # ---- Plot 1: Score distributions ----
    genuine = y_scores[y_true == 1]
    impostor = y_scores[y_true == 0]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(impostor, bins=80, alpha=0.6, color="red", label="Impostor", density=True)
    ax.hist(genuine, bins=80, alpha=0.6, color="green", label="Genuine", density=True)
    ax.set_xlabel("Log-Likelihood Ratio")
    ax.set_ylabel("Density")
    ax.set_title(f"GMM-UBM Score Distribution  (EER = {eer * 100:.2f}%)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.plots_dir / "gmm_ubm_score_dist.png", dpi=150)
    plt.close(fig)

    # ---- Plot 2: DET curve ----
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr * 100, fnr * 100, linewidth=2, color="blue")
    ax.plot([0, 100], [0, 100], "k--", alpha=0.3)
    ax.scatter([eer * 100], [eer * 100], c="red", zorder=5, s=80, label=f"EER = {eer * 100:.2f}%")
    ax.set_xlabel("False Acceptance Rate (%)")
    ax.set_ylabel("False Rejection Rate (%)")
    ax.set_title("GMM-UBM DET Curve")
    ax.set_xlim(0, 50)
    ax.set_ylim(0, 50)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.plots_dir / "gmm_ubm_det.png", dpi=150)
    plt.close(fig)

    print(f"Plots saved to {args.plots_dir}")


if __name__ == "__main__":
    main()
