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
    """Placeholder for MAP adaptation from UBM to target model.

    For scaffolding, we clone the UBM and refit on X to approximate adaptation.
    Replace with proper MAP implementation later.
    """
    base = GaussianMixture(
        n_components=ubm.gmm.n_components,
        covariance_type=ubm.gmm.covariance_type,
        max_iter=ubm.gmm.max_iter,
        random_state=ubm.gmm.random_state,
    )
    base.means_init = ubm.gmm.means_
    base.precisions_init = ubm.gmm.precisions_
    base.weights_init = ubm.gmm.weights_
    base.fit(X)
    return TargetModel(gmm=base)
