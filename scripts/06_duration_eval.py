"""Evaluate both GMM-UBM and LSTM at varying utterance durations.

This is the KEY experiment for the assignment: it shows how each system
degrades as the test utterance gets shorter (0.5s, 1.0s, 1.5s, 2.0s, full).

Produces:
  - results/plots/eer_vs_duration.png  (the headline figure)
  - Console printout of EER at each duration
"""

import sys
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
from src.gmm_ubm import UBMModel


# ---- Helpers ----

def extract_feats_gmm(audio, sr):
    """60-dim features (20 MFCCs + deltas) for the GMM-UBM, returns (T, 60)."""
    return extract_mfcc_features(np.asarray(audio, dtype=np.float32), sr)


def truncate_audio(audio, sr, duration_sec):
    """Truncate audio to duration_sec. Returns None if audio is too short."""
    n_samples = int(duration_sec * sr)
    if len(audio) < n_samples:
        return None
    # Take from the middle for a representative segment
    start = (len(audio) - n_samples) // 2
    return audio[start : start + n_samples]


def compute_eer(y_true, y_scores):
    if len(set(y_true)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    return fpr[idx]


# ---- GMM-UBM scoring at a given duration ----

def eval_gmm_ubm_at_duration(ubm, target_models, test_trials, duration_sec, sr=16000):
    """Score all trials with test audio truncated to duration_sec."""
    y_true, y_scores = [], []

    for test_spk, wav_path in test_trials:
        try:
            audio, file_sr = load_wav(wav_path)
            audio = np.array(audio)
        except Exception:
            continue

        if duration_sec is not None:
            audio = truncate_audio(audio, file_sr, duration_sec)
            if audio is None:
                continue

        try:
            X = extract_feats_gmm(audio, file_sr)
        except Exception:
            continue

        if X.shape[0] < 3:
            continue

        ubm_score = ubm.gmm.score(X)

        for enr_spk, tgt_model in target_models.items():
            tgt_score = tgt_model.score(X)
            llr = tgt_score - ubm_score
            y_scores.append(llr)
            y_true.append(1 if test_spk == enr_spk else 0)

    return compute_eer(np.array(y_true), np.array(y_scores)), len(y_scores)


# ---- LSTM scoring at a given duration ----

def eval_lstm_at_duration(model, device, target_embs, test_trials, duration_sec, sr=16000):
    """Score all trials with test audio truncated to duration_sec."""
    import torch

    y_true, y_scores = [], []

    for test_spk, wav_path in test_trials:
        try:
            audio, file_sr = load_wav(wav_path)
            audio = np.array(audio)
        except Exception:
            continue

        if duration_sec is not None:
            audio = truncate_audio(audio, file_sr, duration_sec)
            if audio is None:
                continue

        try:
            feats = mfcc_with_deltas(audio.astype(np.float32), file_sr)
            feats_t = torch.tensor(feats.T, dtype=torch.float32).unsqueeze(0).to(device)
        except Exception:
            continue

        if feats_t.shape[1] < 3:
            continue

        # Scale
        mean = feats_t.mean(dim=1, keepdim=True)
        std = feats_t.std(dim=1, keepdim=True) + 1e-6
        feats_t = (feats_t - mean) / std

        with torch.no_grad():
            emb = model(feats_t).cpu().numpy().flatten()

        emb = emb / np.linalg.norm(emb)

        for enr_spk, enr_emb in target_embs.items():
            score = float(np.dot(emb, enr_emb))
            y_scores.append(score)
            y_true.append(1 if test_spk == enr_spk else 0)

    return compute_eer(np.array(y_true), np.array(y_scores)), len(y_scores)


# ---- Main ----

def main():
    import torch
    from lstm.model import SpeakerLSTM

    plots_dir = Path("results/plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    durations = [0.1, 0.25, 0.5, 1.0, 1.5, 2.0, None]  # None = full utterance
    duration_labels = ["0.1s", "0.25s", "0.5s", "1.0s", "1.5s", "2.0s", "Full"]

    # ---- Load enrollment list and build test trials ----
    enroll_list = Path("data/lists/test_enrollment_list.txt")
    n_enroll = 3

    enroll_wavs_by_spk = {}
    test_trials = []
    with open(enroll_list) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) != 2:
                continue
            spk_id = Path(parts[0]).name
            wavs = parts[1].split(",")
            enroll_wavs_by_spk[spk_id] = wavs[:n_enroll]
            for wp in wavs[n_enroll:]:
                test_trials.append((spk_id, wp))

    print(f"Test trials: {len(test_trials)} utterances, {len(enroll_wavs_by_spk)} speakers\n")

    # ============================================================
    # GMM-UBM setup
    # ============================================================
    ubm_path = Path("results/models/ubm_model.pkl")
    enrolled_dir = Path("results/enrolled_models")

    gmm_eers = []
    if ubm_path.exists() and enrolled_dir.exists():
        ubm = UBMModel.load(ubm_path)
        target_models = {}
        for pkl in sorted(enrolled_dir.glob("speaker_*.pkl")):
            spk_id = pkl.stem.replace("speaker_", "")
            import pickle
            with open(pkl, 'rb') as pf:
                target_models[spk_id] = pickle.load(pf)
        print(f"GMM-UBM: {ubm.gmm.n_components} components, {len(target_models)} enrolled speakers")

        for dur, label in zip(durations, duration_labels):
            eer, n_trials = eval_gmm_ubm_at_duration(ubm, target_models, test_trials, dur)
            gmm_eers.append(eer * 100)
            print(f"  GMM-UBM @ {label:>5s}: EER = {eer*100:6.2f}%  ({n_trials} trials)")
    else:
        print("GMM-UBM models not found, skipping.")
        gmm_eers = [None] * len(durations)

    print()

    # ============================================================
    # LSTM setup
    # ============================================================
    lstm_path = Path("results/models/lstm/lstm_final.pt")
    lstm_eers = []

    if lstm_path.exists():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(lstm_path, map_location=device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        model = SpeakerLSTM(input_dim=39, hidden_dim=256, num_layers=2,
                            embedding_dim=128, num_classes=None)
        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        model.eval()

        # Compute enrollment d-vectors (using FULL enrollment utterances)
        target_embs = {}
        for spk_id, e_wavs in enroll_wavs_by_spk.items():
            embs = []
            for wp in e_wavs:
                try:
                    audio, sr = load_wav(wp)
                    feats = mfcc_with_deltas(np.array(audio, dtype=np.float32), sr)
                    feats_t = torch.tensor(feats.T, dtype=torch.float32).unsqueeze(0).to(device)
                    mean = feats_t.mean(dim=1, keepdim=True)
                    std = feats_t.std(dim=1, keepdim=True) + 1e-6
                    feats_t = (feats_t - mean) / std
                    with torch.no_grad():
                        emb = model(feats_t).cpu().numpy().flatten()
                    embs.append(emb)
                except Exception:
                    pass
            if embs:
                m = np.mean(embs, axis=0)
                target_embs[spk_id] = m / np.linalg.norm(m)

        print(f"LSTM: {len(target_embs)} enrolled speakers")

        for dur, label in zip(durations, duration_labels):
            eer, n_trials = eval_lstm_at_duration(model, device, target_embs, test_trials, dur)
            lstm_eers.append(eer * 100)
            print(f"  LSTM    @ {label:>5s}: EER = {eer*100:6.2f}%  ({n_trials} trials)")
    else:
        print("LSTM model not found, skipping.")
        lstm_eers = [None] * len(durations)

    # ============================================================
    # Plot: EER vs Duration — GMM-UBM vs Bi-LSTM
    # ============================================================
    print("\n--- Summary ---")
    print(f"{'Duration':<10} {'GMM-UBM':>10} {'Bi-LSTM':>10}")
    print("-" * 32)
    for label, g, l in zip(duration_labels, gmm_eers, lstm_eers):
        gs = f"{g:.2f}%" if g is not None else "N/A"
        ls = f"{l:.2f}%" if l is not None else "N/A"
        print(f"{label:<10} {gs:>10} {ls:>10}")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    x_pos = np.arange(len(duration_labels))

    if any(g is not None for g in gmm_eers):
        valid_g = [(i, v) for i, v in enumerate(gmm_eers) if v is not None]
        ax.plot([x_pos[i] for i, _ in valid_g], [v for _, v in valid_g],
                "o-", color="#4477AA", linewidth=2.5, markersize=8, label="GMM-UBM")

    if any(l is not None for l in lstm_eers):
        valid_l = [(i, v) for i, v in enumerate(lstm_eers) if v is not None]
        ax.plot([x_pos[i] for i, _ in valid_l], [v for _, v in valid_l],
                "s-", color="#EE6677", linewidth=2.5, markersize=8, label="Bi-LSTM")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(duration_labels, fontsize=12)
    ax.set_xlabel("Test Utterance Duration", fontsize=13)
    ax.set_ylabel("Equal Error Rate (%)", fontsize=13)
    ax.set_title("Short-Duration Speaker Verification:\nEER Degradation vs. Utterance Length", fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "eer_vs_duration.png", dpi=150)
    plt.close(fig)
    print(f"\nPlot saved to {plots_dir / 'eer_vs_duration.png'}")


if __name__ == "__main__":
    main()
