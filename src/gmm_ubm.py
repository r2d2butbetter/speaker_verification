"""
Gaussian Mixture Models for UBM training and target enrollment (MAP adaptation).

This file provides minimal class shells to be fleshed out in experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from sklearn.mixture import GaussianMixture


@dataclass
class UBMModel:
    gmm: GaussianMixture

    def save(self, path: str | Path) -> None:
        joblib.dump(self.gmm, path)

    @staticmethod
    def load(path: str | Path) -> "UBMModel":
        gmm = joblib.load(path)
        return UBMModel(gmm=gmm)


def train_ubm(X: np.ndarray, n_components: int = 512, covariance_type: str = "diag", max_iter: int = 100, random_state: int = 0) -> UBMModel:
    """Fit a GMM-UBM on stacked feature frames X: shape (N, D)."""
    gmm = GaussianMixture(n_components=n_components, covariance_type=covariance_type, max_iter=max_iter, random_state=random_state)
    gmm.fit(X)
    return UBMModel(gmm=gmm)


@dataclass
class TargetModel:
    gmm: GaussianMixture

    def score(self, X: np.ndarray) -> float:
        return float(self.gmm.score(X))


def map_adapt(ubm: UBMModel, X: np.ndarray, relevance_factor: float = 16.0) -> TargetModel:
    """MAP adaptation of UBM means toward target speaker data.

    Implements Reynolds et al. (2000) mean-only MAP adaptation:
      1. Compute posterior responsibilities of each UBM component for every frame.
      2. Compute sufficient statistics (zeroth- and first-order).
      3. Blend UBM means with speaker-specific means using the relevance factor.

    Parameters
    ----------
    ubm : UBMModel
        Trained Universal Background Model.
    X : np.ndarray, shape (N, D)
        Feature frames from the target speaker's enrollment utterances.
    relevance_factor : float
        Controls how much weight is given to the UBM prior vs. speaker data.
        Higher = more UBM influence (safer with very little data).
    """
    import copy

    # 1. Compute posterior responsibilities: P(component k | frame x_t)
    posteriors = ubm.gmm.predict_proba(X)          # (N, K)
    # Numerical stability: clip tiny/NaN posteriors
    posteriors = np.nan_to_num(posteriors, nan=0.0, posinf=1.0, neginf=0.0)
    posteriors = np.clip(posteriors, 1e-300, None)

    # 2. Sufficient statistics per component
    n_k = posteriors.sum(axis=0)                    # (K,)  zeroth-order
    F_k = posteriors.T @ X                          # (K, D) first-order
    F_k = np.nan_to_num(F_k, nan=0.0, posinf=0.0, neginf=0.0)

    # 3. MAP-adapted means
    adapted_means = np.copy(ubm.gmm.means_)
    for k in range(ubm.gmm.n_components):
        alpha_k = n_k[k] / (n_k[k] + relevance_factor)
        if n_k[k] > 1e-10:
            speaker_mean_k = F_k[k] / n_k[k]
        else:
            speaker_mean_k = ubm.gmm.means_[k]
        adapted_means[k] = alpha_k * speaker_mean_k + (1 - alpha_k) * ubm.gmm.means_[k]

    # Replace any remaining NaN with UBM means
    nan_mask = ~np.isfinite(adapted_means)
    adapted_means[nan_mask] = ubm.gmm.means_[nan_mask]

    # 4. Build adapted GMM (keep UBM weights and covariances, only change means)
    adapted_gmm = copy.deepcopy(ubm.gmm)
    adapted_gmm.means_ = adapted_means
    return TargetModel(gmm=adapted_gmm)
