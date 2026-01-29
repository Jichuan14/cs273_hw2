"""
HW2 - Problem 2: Cross-Validation

You will implement deterministic K-fold splitting (no shuffling),
and compute CV MSE for polynomial regression.

You may reuse functions from problem1.py (import allowed).
"""

from __future__ import annotations

from typing import List, Tuple, Sequence
import numpy as np

# Reuse pipeline utilities from Problem 1
from problem1 import make_poly_pipeline, mse, predict


def kfold_indices(n: int, K: int) -> List[np.ndarray]:
    """
    Deterministically split indices {0,...,n-1} into K folds in order (no shuffling).

    Return
    ------
    folds : list of length K
        folds[i] is a 1D np.ndarray of validation indices for fold i.
    """
    # TODO: Implement
    raise NotImplementedError


def train_val_split(
    X: np.ndarray, y: np.ndarray, folds: List[np.ndarray], i: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Use folds[i] as validation indices, and the rest as training.
    Return (Xti, yti, Xvi, yvi).
    """
    # TODO: Implement
    raise NotImplementedError


def cv_mse_poly(Xtr: np.ndarray, ytr: np.ndarray, K: int, degree: int = 3) -> float:
    """
    Perform K-fold CV on (Xtr, ytr) for polynomial regression of given degree.

    For each fold:
      - fit polynomial pipeline on training split
      - compute validation MSE
    Return the average validation MSE across folds.
    """
    # TODO: Implement
    raise NotImplementedError


def cv_curve(
    Xtr: np.ndarray, ytr: np.ndarray, degrees: Sequence[int], K: int = 5
) -> np.ndarray:
    """
    Return cv_mses array aligned with degrees.
    """
    # TODO: Implement
    raise NotImplementedError


def recommend_degree_cv(degrees: Sequence[int], cv_mses: np.ndarray) -> int:
    """
    Return the degree with the smallest CV MSE.
    Break ties by returning the smaller degree.
    """
    # TODO: Implement
    raise NotImplementedError
