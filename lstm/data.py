"""
Dataset and DataLoader utilities for PyTorch training on short utterances.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
import librosa
from pathlib import Path
import sys
from collections import defaultdict

# Try to import from src
try:
    from src.features import mfcc_with_deltas
    from src.data_io import get_splits
except ImportError:
    # If run directly from lstm dir
    sys.path.append(str(Path(__file__).parent.parent))
    from src.features import mfcc_with_deltas
    from src.data_io import get_splits


class ShortUtteranceDataset(Dataset):
    def __init__(self, list_of_wavs, speaker2idx, chunk_length_sec=1.5, sr=16000, is_train=True):
        """
        Args:
            list_of_wavs: List of paths to WAV files
            speaker2idx: Dictionary mapping speaker string to integer ID
            chunk_length_sec: Target length for audio chunks
            sr: Sampling rate
            is_train: If True, yields triplets (Anchor, Positive, Negative). If False, processes normally.
        """
        self.wavs = list_of_wavs
        self.speaker2idx = speaker2idx
        self.chunk_length_sec = chunk_length_sec
        self.chunk_samples = int(chunk_length_sec * sr)
        self.sr = sr
        self.is_train = is_train
        
        # Build speaker to wav mappings for triplet generation
        if self.is_train:
            self.speaker2wavs = defaultdict(list)
            for w in self.wavs:
                spk = Path(w).parent.name
                self.speaker2wavs[spk].append(w)
            self.speakers = list(self.speaker2wavs.keys())

    def __len__(self):
        return len(self.wavs)

    def extract_features(self, y):
        """Extract MFCC+deltas and transpose to (time, features) for PyTorch RNNs"""
        feats = mfcc_with_deltas(y, self.sr)
        # librosa returns (features, time). PyTorch LSTM expects (batch, time, features)
        return torch.tensor(feats.T, dtype=torch.float32)

    def load_chunk(self, wav_path):
        y, sr = librosa.load(str(wav_path), sr=self.sr)
        
        if self.is_train:
            # If audio is shorter than expected, pad it
            if len(y) < self.chunk_samples:
                pad_len = self.chunk_samples - len(y)
                y = np.pad(y, (0, pad_len), mode='constant')
                
            # Take a random chunk limit for short-duration training
            if len(y) > self.chunk_samples:
                start = np.random.randint(0, len(y) - self.chunk_samples)
                y = y[start:start + self.chunk_samples]
        else:
            # For evaluation, use the full utterance. (No padding, no cropping needed)
            pass
            
        return self.extract_features(y)

    def __getitem__(self, idx):
        if not self.is_train:
            wav_path = Path(self.wavs[idx])
            speaker_id = wav_path.parent.name
            label = self.speaker2idx[speaker_id]
            feats = self.load_chunk(wav_path)
            return feats, torch.tensor(label, dtype=torch.long)
            
        # Triplet Generation Logic
        anchor_path = self.wavs[idx]
        anchor_spk = Path(anchor_path).parent.name
        
        # Get Positive (same speaker, different utterance if possible)
        pos_candidates = self.speaker2wavs[anchor_spk]
        if len(pos_candidates) > 1:
            pos_path = np.random.choice([p for p in pos_candidates if p != anchor_path])
        else:
            pos_path = anchor_path # Fallback if only 1 utterance
            
        # Get Negative (different speaker)
        neg_spk = np.random.choice([s for s in self.speakers if s != anchor_spk])
        neg_path = np.random.choice(self.speaker2wavs[neg_spk])
        
        anchor_feats = self.load_chunk(anchor_path)
        pos_feats = self.load_chunk(pos_path)
        neg_feats = self.load_chunk(neg_path)
        
        return anchor_feats, pos_feats, neg_feats
