"""Compare GMM-UBM vs LSTM speaker verification results side-by-side.

Run AFTER:
  - scripts/04_run_short_eval.py  (produces results/scores/gmm_ubm_scores.npz)
  - lstm/enroll_and_eval.py        (produces LSTM scores on the fly)

This script generates:
  1. Overlaid DET curves
  2. EER comparison bar chart
  3. Score distribution panels (one per system)
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from collections import defaultdict

sys.path.append(str(Path(__file__).parent.parent))


def compute_eer(y_true, y_scores):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    return fpr[idx], fpr, fnr


def get_lstm_scores():
    """Reproduce the LSTM scoring pipeline and return (y_true, y_scores)."""
    import torch
    from lstm.model import SpeakerLSTM
    from lstm.data import ShortUtteranceDataset

    lists_dir = Path("data/lists")
    enroll_list_path = lists_dir / "test_enrollment_list.txt"

    enroll_wavs, test_wavs = [], []
    with open(enroll_list_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) == 2:
                paths = parts[1].split(",")
                if len(paths) >= 3:
                    enroll_wavs.extend(paths[:3])
                    test_wavs.extend(paths[3:])
                else:
                    enroll_wavs.extend(paths)

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

    dummy_dict = defaultdict(lambda: 0)

    def extract_embs(wav_list):
        ds = ShortUtteranceDataset(wav_list, dummy_dict, chunk_length_sec=1.5, is_train=False)
        speaker_embs = defaultdict(list)
        with torch.no_grad():
            for i in range(len(ds)):
                feats, _ = ds[i]
                feats = feats.unsqueeze(0).to(device)
                mean = feats.mean(dim=1, keepdim=True)
                std = feats.std(dim=1, keepdim=True) + 1e-6
                feats = (feats - mean) / std
                emb = model(feats)
                spk = Path(ds.wavs[i]).parent.name
                speaker_embs[spk].append(emb.cpu().numpy().flatten())
        return speaker_embs

    enroll_embs = extract_embs(enroll_wavs)
    target_models = {}
    for spk, embs in enroll_embs.items():
        m = np.mean(embs, axis=0)
        target_models[spk] = m / np.linalg.norm(m)

    test_embs = extract_embs(test_wavs)

    y_true, y_scores = [], []
    for test_spk, t_embs in test_embs.items():
        for t_emb in t_embs:
            t_norm = t_emb / np.linalg.norm(t_emb)
            for enr_spk, enr_emb in target_models.items():
                y_scores.append(float(np.dot(t_norm, enr_emb)))
                y_true.append(1 if test_spk == enr_spk else 0)

    return np.array(y_true), np.array(y_scores)


def main():
    plots_dir = Path("results/plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load GMM-UBM scores ----
    gmm_scores_path = Path("results/scores/gmm_ubm_scores.npz")
    if not gmm_scores_path.exists():
        print("GMM-UBM scores not found. Run scripts/04_run_short_eval.py first.")
        return
    gmm_data = np.load(gmm_scores_path)
    gmm_true, gmm_scores = gmm_data["y_true"], gmm_data["y_scores"]
    gmm_eer, gmm_fpr, gmm_fnr = compute_eer(gmm_true, gmm_scores)
    print(f"GMM-UBM  EER: {gmm_eer * 100:.2f}%  ({len(gmm_scores)} trials)")

    # ---- Compute LSTM scores ----
    lstm_path = Path("results/models/lstm/lstm_final.pt")
    if not lstm_path.exists():
        print("LSTM model not found. Train it first.")
        return
    print("Computing LSTM scores...")
    lstm_true, lstm_scores = get_lstm_scores()
    lstm_eer, lstm_fpr, lstm_fnr = compute_eer(lstm_true, lstm_scores)
    print(f"LSTM     EER: {lstm_eer * 100:.2f}%  ({len(lstm_scores)} trials)")

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
