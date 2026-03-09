"""
Feature extraction: MFCC + delta + delta-delta.

This is a minimal scaffold; tune parameters in experiments.
"""

from typing import Tuple
import numpy as np
import librosa


def mfcc_with_deltas(y: np.ndarray, sr: int, n_mfcc: int = 13, n_fft: int = 400, hop_length: int = 160) -> np.ndarray:
    """Compute MFCCs with deltas and delta-deltas.

    Returns array of shape (3*n_mfcc, T).
    """
    y = np.asarray(y, dtype=np.float32)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    d1 = librosa.feature.delta(mfcc)
    d2 = librosa.feature.delta(mfcc, order=2)
    feats = np.vstack([mfcc, d1, d2])
    return feats
