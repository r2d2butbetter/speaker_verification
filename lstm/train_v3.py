"""Improved LSTM training v3 — keeps the working triplet generation,
adds data augmentation + shorter/variable chunks.

Key changes vs original train.py:
  1. Random chunk length between 0.4s-1.2s (was fixed 1.5s)
  2. Additive noise augmentation (SNR 10-25 dB)
  3. Time-masking augmentation (SpecAugment-lite)
  4. Smaller margin (0.3) — better for L2-normalized embeddings
  5. Cosine annealing LR, 40 epochs, early stopping

Run from speaker_verification/:
    python -m lstm.train_v3
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import librosa
from collections import defaultdict

sys.path.append(str(Path(__file__).parent.parent))

from src.features import mfcc_with_deltas
from lstm.model import SpeakerLSTM


# ─── Augmentation ────────────────────────────────────────────────

def add_noise(y, snr_db_range=(10, 25)):
    snr_db = np.random.uniform(*snr_db_range)
    sig_power = np.mean(y ** 2) + 1e-10
    noise_power = sig_power / (10 ** (snr_db / 10))
    return y + np.random.normal(0, np.sqrt(noise_power), len(y)).astype(y.dtype)


def time_mask(feats, max_frames=5):
    T = feats.shape[0]
    if T <= max_frames + 2:
        return feats
    mask_len = np.random.randint(1, min(max_frames, T // 4) + 1)
    start = np.random.randint(0, T - mask_len)
    feats = feats.copy()
    feats[start:start + mask_len, :] = 0.0
    return feats


# ─── Dataset: triplet generation + augmentation + variable chunks ─

class AugTripletDataset(Dataset):
    def __init__(self, wav_list, sr=16000,
                 min_chunk_sec=0.4, max_chunk_sec=1.2, augment=True):
        self.wavs = wav_list
        self.sr = sr
        self.min_samples = int(min_chunk_sec * sr)
        self.max_samples = int(max_chunk_sec * sr)
        self.augment = augment

        self.speaker2wavs = defaultdict(list)
        for w in self.wavs:
            spk = Path(w).parent.name
            self.speaker2wavs[spk].append(w)
        self.speakers = list(self.speaker2wavs.keys())

    def __len__(self):
        return len(self.wavs)

    def _load_chunk(self, wav_path):
        y, _ = librosa.load(str(wav_path), sr=self.sr)

        chunk_len = np.random.randint(self.min_samples, self.max_samples + 1)
        if len(y) < chunk_len:
            y = np.pad(y, (0, chunk_len - len(y)), mode='constant')
        start = np.random.randint(0, max(1, len(y) - chunk_len))
        y = y[start:start + chunk_len]

        if self.augment and np.random.rand() < 0.4:
            y = add_noise(y)

        feats = mfcc_with_deltas(y.astype(np.float32), self.sr)  # (39, T)
        feats = feats.T  # (T, 39)

        if self.augment and np.random.rand() < 0.25:
            feats = time_mask(feats)

        return torch.tensor(feats, dtype=torch.float32)

    def __getitem__(self, idx):
        anchor_path = self.wavs[idx]
        anchor_spk = Path(anchor_path).parent.name

        # Positive: same speaker, different utterance
        pos_candidates = self.speaker2wavs[anchor_spk]
        if len(pos_candidates) > 1:
            pos_path = np.random.choice([p for p in pos_candidates if p != anchor_path])
        else:
            pos_path = anchor_path

        # Negative: different speaker
        neg_spk = np.random.choice([s for s in self.speakers if s != anchor_spk])
        neg_path = np.random.choice(self.speaker2wavs[neg_spk])

        return self._load_chunk(anchor_path), self._load_chunk(pos_path), self._load_chunk(neg_path)


def collate_triplets(batch):
    """Pad each of anchor/positive/negative to max length in the batch."""
    anchors, positives, negatives = zip(*batch)

    def pad_batch(tensors):
        max_len = max(t.shape[0] for t in tensors)
        padded = torch.zeros(len(tensors), max_len, tensors[0].shape[1])
        for i, t in enumerate(tensors):
            padded[i, :t.shape[0], :] = t
        return padded

    return pad_batch(anchors), pad_batch(positives), pad_batch(negatives)


# ─── Training ────────────────────────────────────────────────────

def train():
    lists_dir = Path("data/lists")
    with open(lists_dir / "ubm_train_list.txt") as f:
        train_wavs = [l.strip() for l in f if l.strip()]

    if not train_wavs:
        print("No training wavs. Run scripts/01_prep_timit.py first.")
        return

    print(f"Training on {len(train_wavs)} files, "
          f"{len(set(Path(w).parent.name for w in train_wavs))} speakers.")

    ds = AugTripletDataset(train_wavs, min_chunk_sec=0.4, max_chunk_sec=1.2, augment=True)
    loader = DataLoader(ds, batch_size=32, shuffle=True, num_workers=0,
                        collate_fn=collate_triplets)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = SpeakerLSTM(input_dim=39, hidden_dim=256, num_layers=2,
                        embedding_dim=128, num_classes=None).to(device)

    criterion = nn.TripletMarginLoss(margin=0.3, p=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40, eta_min=1e-5)

    num_epochs = 40
    best_loss = float("inf")
    patience, patience_counter = 10, 0

    models_dir = Path("results/models/lstm_v3")
    models_dir.mkdir(parents=True, exist_ok=True)

    def scale(f):
        return (f - f.mean(dim=1, keepdim=True)) / (f.std(dim=1, keepdim=True) + 1e-6)

    print("Starting training...")
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for anchor, pos, neg in loader:
            anchor = scale(anchor).to(device)
            pos = scale(pos).to(device)
            neg = scale(neg).to(device)

            optimizer.zero_grad()
            a_emb = model(anchor)
            p_emb = model(pos)
            n_emb = model(neg)

            loss = criterion(a_emb, p_emb, n_emb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        epoch_loss = total_loss / max(n_batches, 1)
        lr_now = scheduler.get_last_lr()[0]
        scheduler.step()

        print(f"  Epoch {epoch:>2d}/{num_epochs} | Loss: {epoch_loss:.4f} | LR: {lr_now:.6f}")

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "loss": epoch_loss,
        }, models_dir / f"lstm_v3_epoch_{epoch}.pt")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience_counter = 0
            torch.save(model.state_dict(), models_dir / "lstm_v3_best.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch}.")
                break

    torch.save(model.state_dict(), models_dir / "lstm_v3_final.pt")
    print(f"Done. Best loss: {best_loss:.4f}. Models in {models_dir}")


if __name__ == "__main__":
    train()
