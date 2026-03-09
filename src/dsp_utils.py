"""
DSP utilities: simple VAD and truncation helpers for short-duration experiments.
"""

from typing import Tuple
import numpy as np


def energy_vad(y: np.ndarray, frame_length: int, hop_length: int, thresh: float = 0.1) -> np.ndarray:
    """Simple frame-energy-based VAD mask over samples.

    Returns a boolean mask per-sample indicating voiced regions.
    """
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    y = np.asarray(y, dtype=np.float32)
    n_frames = max(1, 1 + (len(y) - frame_length) // hop_length)
    energies = np.empty(n_frames, dtype=np.float32)
    for i in range(n_frames):
        s = i * hop_length
        e = s + frame_length
        frame = y[s:e] if e <= len(y) else y[s:]
        energies[i] = np.mean(frame * frame) if len(frame) else 0.0
    m = energies > (thresh * np.max(energies) if energies.size else 0.0)
    # Expand to sample-level mask
    mask = np.zeros_like(y, dtype=bool)
    for i, keep in enumerate(m):
        s = i * hop_length
        e = min(s + frame_length, len(y))
        if keep:
            mask[s:e] = True
    return mask


def truncate_audio(y: np.ndarray, sr: int, max_duration_s: float) -> np.ndarray:
    """Truncate to max duration (seconds)."""
    max_samples = int(round(max_duration_s * sr))
    return y[:max_samples]
