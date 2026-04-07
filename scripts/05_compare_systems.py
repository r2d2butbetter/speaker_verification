"""Compare GMM-UBM vs LSTM speaker verification results side-by-side.

Run AFTER:
  - scripts/03_enroll_targets.py   (enrolls GMM speaker models)
  - lstm/train.py                  (trains the LSTM model)

Both systems are evaluated on the same test enrollment list with the same
trial protocol (enroll on first 3 utterances, test on the rest, all-vs-all).

This script generates:
  1. Overlaid DET curves
  2. EER comparison bar chart
  3. Score distribution panels (one per system)
"""

import sys
import pickle
import random
from pathlib import Path
import numpy as np
import librosa
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from collections import defaultdict

sys.path.append(str(Path(__file__).parent.parent))

from src.data_io import load_wav
from src.features import extract_mfcc_features, mfcc_with_deltas
from src.dsp_utils import extract_longest_word


def compute_eer(y_true, y_scores):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    return fpr[idx], fpr, fnr


def parse_enrollment_list(list_path, n_enroll=3):
    """Parse test_enrollment_list.txt → per-speaker enroll/test wavs."""
    enroll_by_spk, test_trials = {}, []
    with open(list_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) != 2:
                continue
            spk_id = Path(parts[0]).name
            wavs = parts[1].split(",")
            enroll_by_spk[spk_id] = wavs[:n_enroll]
            for wp in wavs[n_enroll:]:
                test_trials.append((spk_id, wp))
    return enroll_by_spk, test_trials


def get_gmm_scores():
    """Score GMM-UBM on wake-word segments (<0.8s), matching 04_run_short_eval.

    Uses SI test files, extracts the longest word from each via .WRD,
    scores true trial + 3 random impostors per utterance.
    Returns (y_true, y_scores).
    """
    ubm_path = Path("results/models/ubm_model.pkl")
    enrolled_dir = Path("results/enrolled_models")
    list_path = Path("data/lists/test_enrollment_list.txt")

    ubm = joblib.load(ubm_path)

    # Load enrolled speaker models and parse test list
    with open(list_path) as f:
        lines = f.read().splitlines()

    test_data = {}
    target_models = {}
    for line in lines:
        speaker_id, paths_str = line.split("|")
        clean_id = Path(speaker_id).name
        test_paths = [p for p in paths_str.split(",") if "SI" in Path(p).name]
        test_data[clean_id] = test_paths
        model_path = enrolled_dir / f"speaker_{clean_id}.pkl"
        if model_path.exists():
            with open(model_path, "rb") as pf:
                target_models[clean_id] = pickle.load(pf)

    speaker_ids = list(target_models.keys())
    true_scores, impostor_scores = [], []

    for target_id in speaker_ids:
        target_gmm = target_models[target_id]
        for wav_path_str in test_data.get(target_id, []):
            wav_path = Path(wav_path_str)
            wrd_path = wav_path.with_suffix("").with_suffix(".WRD")
            try:
                audio, sr, word, duration = extract_longest_word(wav_path, wrd_path)
                if len(audio) < int(sr * 0.1):
                    continue
                features = extract_mfcc_features(audio, sr)
                score_ubm = ubm.score(features)

                # True trial
                llr_true = target_gmm.score(features) - score_ubm
                true_scores.append(llr_true)

                # 3 random impostor trials
                impostors = random.sample([s for s in speaker_ids if s != target_id], 3)
                for imp_id in impostors:
                    llr_imp = target_models[imp_id].score(features) - score_ubm
                    impostor_scores.append(llr_imp)
            except Exception:
                continue

    y_true = np.array([1] * len(true_scores) + [0] * len(impostor_scores))
    y_scores = np.array(true_scores + impostor_scores)
    return y_true, y_scores


def get_lstm_scores(enroll_by_spk, test_trials):
    """Score all trials with Bi-LSTM (cosine sim). Returns (y_true, y_scores)."""
    import torch
    from lstm.model import SpeakerLSTM

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = Path("results/models/lstm/lstm_final.pt")
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model = SpeakerLSTM(input_dim=39, hidden_dim=256, num_layers=2,
                        embedding_dim=128, num_classes=None)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    def wav_to_emb(wav_path):
        audio, sr = load_wav(wav_path)
        feats = mfcc_with_deltas(np.array(audio, dtype=np.float32), sr)
        feats_t = torch.tensor(feats.T, dtype=torch.float32).unsqueeze(0).to(device)
        mean = feats_t.mean(dim=1, keepdim=True)
        std = feats_t.std(dim=1, keepdim=True) + 1e-6
        feats_t = (feats_t - mean) / std
        with torch.no_grad():
            emb = model(feats_t).cpu().numpy().flatten()
        return emb

    # Enroll
    target_embs = {}
    for spk_id, e_wavs in enroll_by_spk.items():
        embs = []
        for wp in e_wavs:
            try:
                embs.append(wav_to_emb(wp))
            except Exception:
                pass
        if embs:
            m = np.mean(embs, axis=0)
            target_embs[spk_id] = m / np.linalg.norm(m)

    # Test
    y_true, y_scores = [], []
    for test_spk, wav_path in test_trials:
        try:
            emb = wav_to_emb(wav_path)
        except Exception:
            continue
        emb = emb / np.linalg.norm(emb)
        for enr_spk, enr_emb in target_embs.items():
            y_scores.append(float(np.dot(emb, enr_emb)))
            y_true.append(1 if test_spk == enr_spk else 0)

    return np.array(y_true), np.array(y_scores)


def main():
    plots_dir = Path("results/plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    enroll_list = Path("data/lists/test_enrollment_list.txt")
    enroll_by_spk, test_trials = parse_enrollment_list(enroll_list)
    print(f"Test trials: {len(test_trials)} utterances, {len(enroll_by_spk)} speakers\n")

    # ---- GMM-UBM scores (wake-word <0.8s, matching 04_run_short_eval) ----
    print("Computing GMM-UBM scores (wake-word segments)...")
    gmm_true, gmm_scores = get_gmm_scores()
    gmm_eer, gmm_fpr, gmm_fnr = compute_eer(gmm_true, gmm_scores)
    print(f"GMM-UBM  EER: {gmm_eer * 100:.2f}%  ({len(gmm_scores)} trials)")

    # ---- LSTM scores ----
    print("Computing LSTM scores...")
    lstm_true, lstm_scores = get_lstm_scores(enroll_by_spk, test_trials)
    lstm_eer, lstm_fpr, lstm_fnr = compute_eer(lstm_true, lstm_scores)
    print(f"Bi-LSTM  EER: {lstm_eer * 100:.2f}%  ({len(lstm_scores)} trials)")

    # ===== PLOT 1: Overlaid DET curves =====
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(gmm_fpr * 100, gmm_fnr * 100, linewidth=2, color="blue",
            label=f"GMM-UBM (EER={gmm_eer*100:.2f}%)")
    ax.plot(lstm_fpr * 100, lstm_fnr * 100, linewidth=2, color="green",
            label=f"Bi-LSTM (EER={lstm_eer*100:.2f}%)")
    ax.plot([0, 50], [0, 50], "k--", alpha=0.3)
    ax.scatter([gmm_eer*100], [gmm_eer*100], c="blue", zorder=5, s=80)
    ax.scatter([lstm_eer*100], [lstm_eer*100], c="green", zorder=5, s=80)
    ax.set_xlabel("False Acceptance Rate (%)", fontsize=12)
    ax.set_ylabel("False Rejection Rate (%)", fontsize=12)
    ax.set_title("DET Curve: GMM-UBM vs Bi-LSTM\n(Short-Duration Speaker Verification)", fontsize=13)
    ax.set_xlim(0, 50)
    ax.set_ylim(0, 50)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "comparison_det.png", dpi=150)
    plt.close(fig)

    # ===== PLOT 2: EER bar chart =====
    fig, ax = plt.subplots(figsize=(5, 4))
    systems = ["GMM-UBM", "Bi-LSTM"]
    eers = [gmm_eer * 100, lstm_eer * 100]
    colors = ["#4477AA", "#228833"]
    bars = ax.bar(systems, eers, color=colors, edgecolor="black", width=0.5)
    for bar, val in zip(bars, eers):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.2f}%", ha="center", va="bottom", fontweight="bold", fontsize=12)
    ax.set_ylabel("Equal Error Rate (%)", fontsize=12)
    ax.set_title("EER Comparison: Short-Duration Speaker Verification", fontsize=13)
    ax.set_ylim(0, max(eers) * 1.3)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "comparison_eer_bar.png", dpi=150)
    plt.close(fig)

    # ===== PLOT 3: Score distributions side by side =====
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    for ax, (label, yt, ys, eer, color) in zip(axes, [
        ("GMM-UBM (LLR)", gmm_true, gmm_scores, gmm_eer, "blue"),
        ("Bi-LSTM (Cosine Sim.)", lstm_true, lstm_scores, lstm_eer, "green"),
    ]):
        genuine = ys[yt == 1]
        impostor = ys[yt == 0]
        ax.hist(impostor, bins=80, alpha=0.5, color="red", label="Impostor", density=True)
        ax.hist(genuine, bins=80, alpha=0.5, color=color, label="Genuine", density=True)
        ax.set_title(f"{label}  (EER={eer*100:.2f}%)", fontsize=12)
        ax.set_xlabel("Score")
        ax.set_ylabel("Density")
        ax.legend()

    fig.suptitle("Score Distributions: Genuine vs Impostor", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(plots_dir / "comparison_score_dist.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nAll comparison plots saved to {plots_dir}/")
    print(f"  - comparison_det.png")
    print(f"  - comparison_eer_bar.png")
    print(f"  - comparison_score_dist.png")


if __name__ == "__main__":
    main()
