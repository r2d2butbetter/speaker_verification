"""Improved LSTM training for short-duration speaker verification.

Key improvements over train.py:
  1. Data augmentation (additive noise, time masking)
  2. Online semi-hard negative mining within each batch
  3. Multi-scale chunk lengths (0.3s - 1.5s randomly per sample)
  4. Cosine annealing LR schedule for smoother convergence
  5. More epochs (40) with early stopping patience

Run from speaker_verification/:
    python -m lstm.train_v2
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import librosa
from collections import defaultdict

sys.path.append(str(Path(__file__).parent.parent))

from src.features import mfcc_with_deltas
from lstm.model import SpeakerLSTM


# ─── Data augmentation ─────────────────────────────────────────────

def add_noise(y, snr_db_range=(5, 20)):
    """Add white Gaussian noise at a random SNR."""
    snr_db = np.random.uniform(*snr_db_range)
    signal_power = np.mean(y ** 2) + 1e-10
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), len(y))
    return y + noise.astype(y.dtype)


def time_mask(feats, max_mask_frames=8):
    """Zero out a random contiguous block of frames (SpecAugment-lite)."""
    T = feats.shape[0]
    if T <= max_mask_frames:
        return feats
    mask_len = np.random.randint(1, min(max_mask_frames, T // 3) + 1)
    start = np.random.randint(0, T - mask_len)
    feats = feats.copy()
    feats[start:start + mask_len, :] = 0.0
    return feats


# ─── Improved dataset with multi-scale chunks + augmentation ──────

class AugmentedDataset(Dataset):
    """Yields (features, speaker_label) with random chunk lengths and augmentation."""

    def __init__(self, wav_list, speaker2idx, sr=16000,
                 min_chunk_sec=0.3, max_chunk_sec=1.5, augment=True):
        self.wavs = wav_list
        self.speaker2idx = speaker2idx
        self.sr = sr
        self.min_chunk = int(min_chunk_sec * sr)
        self.max_chunk = int(max_chunk_sec * sr)
        self.augment = augment

        self.speaker2wavs = defaultdict(list)
        for w in self.wavs:
            spk = Path(w).parent.name
            self.speaker2wavs[spk].append(w)
        self.speakers = list(self.speaker2wavs.keys())

    def __len__(self):
        return len(self.wavs)

    def _load_random_chunk(self, wav_path):
        y, _ = librosa.load(str(wav_path), sr=self.sr)

        # Random chunk length between min and max
        chunk_len = np.random.randint(self.min_chunk, self.max_chunk + 1)

        if len(y) < chunk_len:
            y = np.pad(y, (0, chunk_len - len(y)), mode='constant')

        start = np.random.randint(0, max(1, len(y) - chunk_len))
        y = y[start:start + chunk_len]

        if self.augment:
            # 50% chance of noise
            if np.random.rand() < 0.5:
                y = add_noise(y)

        feats = mfcc_with_deltas(y.astype(np.float32), self.sr)
        feats = feats.T  # (T, 39)

        if self.augment:
            # 30% chance of time masking
            if np.random.rand() < 0.3:
                feats = time_mask(feats)

        return torch.tensor(feats, dtype=torch.float32)

    def __getitem__(self, idx):
        wav_path = self.wavs[idx]
        spk = Path(wav_path).parent.name
        label = self.speaker2idx[spk]
        feats = self._load_random_chunk(wav_path)
        return feats, torch.tensor(label, dtype=torch.long)


def collate_pad(batch):
    """Pad variable-length feature sequences to the longest in the batch."""
    feats_list, labels = zip(*batch)
    max_len = max(f.shape[0] for f in feats_list)
    padded = torch.zeros(len(feats_list), max_len, feats_list[0].shape[1])
    for i, f in enumerate(feats_list):
        padded[i, :f.shape[0], :] = f
    return padded, torch.stack(labels)


# ─── Online semi-hard triplet loss ───────────────────────────────

def batch_all_triplet_loss(embeddings, labels, margin=0.3):
    """Compute triplet loss with online semi-hard mining over the batch.

    For every (anchor, positive, negative) triplet in the batch where
    anchor and positive share the same label and negative doesn't,
    compute loss = max(0, d(a,p) - d(a,n) + margin).
    Uses only semi-hard triplets: d(a,p) < d(a,n) < d(a,p) + margin.
    """
    # Pairwise distance matrix
    dist_mat = torch.cdist(embeddings, embeddings, p=2)  # (B, B)

    B = embeddings.size(0)
    labels = labels.view(B)

    # Masks
    same = labels.unsqueeze(0) == labels.unsqueeze(1)      # (B, B)
    diff = ~same

    total_loss = torch.tensor(0.0, device=embeddings.device)
    n_triplets = 0

    for a in range(B):
        pos_mask = same[a].clone()
        pos_mask[a] = False  # exclude self
        neg_mask = diff[a]

        if pos_mask.sum() == 0 or neg_mask.sum() == 0:
            continue

        # For each anchor, pick the hardest positive
        ap_dists = dist_mat[a][pos_mask]
        hardest_pos_dist = ap_dists.max()

        # Semi-hard negatives: d(a,n) > d(a,p) but d(a,n) < d(a,p) + margin
        an_dists = dist_mat[a][neg_mask]
        semi_hard = an_dists[(an_dists > hardest_pos_dist) &
                             (an_dists < hardest_pos_dist + margin)]

        if len(semi_hard) > 0:
            hardest_neg_dist = semi_hard.min()
        else:
            # Fallback: use the closest negative (hard negative)
            hardest_neg_dist = an_dists.min()

        loss = F.relu(hardest_pos_dist - hardest_neg_dist + margin)
        total_loss += loss
        n_triplets += 1

    if n_triplets == 0:
        return total_loss
    return total_loss / n_triplets


# ─── Training ────────────────────────────────────────────────────

def train():
    lists_dir = Path("data/lists")
    ubm_list_path = lists_dir / "ubm_train_list.txt"

    with open(ubm_list_path) as f:
        train_wavs = [l.strip() for l in f if l.strip()]

    if not train_wavs:
        print("No training wavs. Run scripts/01_prep_timit.py first.")
        return

    speakers = sorted(set(Path(w).parent.name for w in train_wavs))
    speaker2idx = {s: i for i, s in enumerate(speakers)}
    print(f"Training on {len(train_wavs)} files, {len(speakers)} speakers.")

    # Dataset with multi-scale chunks (0.3s – 1.5s) and augmentation
    ds = AugmentedDataset(train_wavs, speaker2idx,
                          min_chunk_sec=0.3, max_chunk_sec=1.5, augment=True)

    # Larger batch so the online miner has more triplets to choose from
    loader = DataLoader(ds, batch_size=64, shuffle=True, num_workers=0,
                        collate_fn=collate_pad)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = SpeakerLSTM(input_dim=39, hidden_dim=256, num_layers=2,
                        embedding_dim=128, num_classes=None).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40, eta_min=1e-5)

    num_epochs = 40
    margin = 0.3
    best_loss = float("inf")
    patience, patience_counter = 8, 0

    models_dir = Path("results/models/lstm_v2")
    models_dir.mkdir(parents=True, exist_ok=True)

    print("Starting improved training...")
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0

        for batch_idx, (feats, labels) in enumerate(loader):
            # Instance normalization
            mean = feats.mean(dim=1, keepdim=True)
            std = feats.std(dim=1, keepdim=True) + 1e-6
            feats = (feats - mean) / std
            feats, labels = feats.to(device), labels.to(device)

            optimizer.zero_grad()
            embeddings = model(feats)

            loss = batch_all_triplet_loss(embeddings, labels, margin=margin)
            if loss.item() == 0.0:
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += loss.item()

        epoch_loss = total_loss / max(len(loader), 1)
        lr_now = scheduler.get_last_lr()[0]
        scheduler.step()

        print(f"  Epoch {epoch:>2d}/{num_epochs} | Loss: {epoch_loss:.4f} | LR: {lr_now:.6f}")

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": epoch_loss,
            "speaker2idx": speaker2idx,
        }, models_dir / f"lstm_v2_epoch_{epoch}.pt")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience_counter = 0
            torch.save(model.state_dict(), models_dir / "lstm_v2_best.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch} (no improvement for {patience} epochs).")
                break

    torch.save(model.state_dict(), models_dir / "lstm_v2_final.pt")
    print(f"Done. Best loss: {best_loss:.4f}. Models saved to {models_dir}")


if __name__ == "__main__":
    train()
