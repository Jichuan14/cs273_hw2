"""
HW2 - Problem 1: Linear/Polynomial Regression + Model Selection

You may use: numpy, pandas, scikit-learn.
Do NOT use any course-bundled / legacy ML libraries.

All functions must keep the same names/signatures.
The autograder will import and call these functions directly.
"""

from __future__ import annotations

from typing import Tuple, Sequence, Dict, Any
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


def load_curve80(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load curve80.txt where column 0 is x and column 1 is y.

    Returns
    -------
    X : np.ndarray of shape (N, 1)
    y : np.ndarray of shape (N,)
    """
    # TODO: Implement loading
    data = np.loadtxt(path)
    X = data[:, 0].reshape(-1, 1)
    y = data[:, 1]
    return X, y


def split_data(
    X: np.ndarray, y: np.ndarray, frac: float = 0.75
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Deterministically split without shuffling:
    first frac portion for train, remaining for test.
    """
    # TODO: Implement deterministic split
    n_train = int(frac * len(X))
    X_train = X[:n_train]
    y_train = y[:n_train]
    X_test = X[n_train:]
    y_test = y[n_train:]
    return X_train, X_test, y_train, y_test


def shapes(
    Xtr: np.ndarray, Xte: np.ndarray, ytr: np.ndarray, yte: np.ndarray
) -> Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]:
    """
    Return shapes as tuples: (Xtr.shape, Xte.shape, ytr.shape, yte.shape)
    """
    # TODO: Implement
    return Xtr.shape, Xte.shape, ytr.shape, yte.shape


def fit_linear(Xtr: np.ndarray, ytr: np.ndarray) -> LinearRegression:
    """
    Fit a baseline linear regression model with intercept.
    """
    # TODO: Implement training
    model = LinearRegression(fit_intercept=True)
    model.fit(Xtr, ytr)
    return model


def predict(model: Any, X: np.ndarray) -> np.ndarray:
    """
    Return predictions as a 1D array of length N.
    """
    # TODO: Implement prediction wrapper
    pred = model.predict(X)
    return pred


def mse(yhat: np.ndarray, y: np.ndarray) -> float:
    """
    Mean squared error.
    """
    # TODO: Implement MSE
    err = yhat - y
    return np.mean(err ** 2)


def eval_linear(
    Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, yte: np.ndarray
) -> Tuple[float, float]:
    """
    Train baseline linear regression, return (train_mse, test_mse).
    """
    # TODO: Implement evaluation
    model = fit_linear(Xtr, ytr)
    yhat = predict(model, Xte)
    train_mse = mse(model.predict(Xtr), ytr)
    test_mse = mse(yhat, yte)
    return train_mse, test_mse


def make_poly_pipeline(degree: int = 3) -> Pipeline:
    """
    Create a scikit-learn pipeline:
      PolynomialFeatures(degree, include_bias=False)
      StandardScaler()
      LinearRegression(fit_intercept=True)

    Returns an *unfitted* Pipeline.
    """
    # TODO: Create and return pipeline
    pipeline = Pipeline([
        ('poly', PolynomialFeatures(degree, include_bias=False)),
        ('scaler', StandardScaler()),
        ('linear', LinearRegression(fit_intercept=True))
    ])
    return pipeline


def fit_poly(Xtr: np.ndarray, ytr: np.ndarray, degree: int) -> Pipeline:
    """
    Fit the polynomial pipeline of the given degree on training data.
    """
    # TODO: Fit and return pipeline
    model = make_poly_pipeline(degree)
    model.fit(Xtr, ytr)
    return model


def eval_poly(
    Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, yte: np.ndarray, degree: int = 3
) -> Tuple[float, float]:
    """
    Fit polynomial regression of given degree and return (train_mse, test_mse).
    """
    # TODO: Implement
    model = fit_poly(Xtr, ytr, degree)
    yhat = predict(model, Xte)
    train_mse = mse(model.predict(Xtr), ytr)
    test_mse = mse(yhat, yte)
    return train_mse, test_mse


def eval_degrees(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xte: np.ndarray,
    yte: np.ndarray,
    degrees: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluate a list/tuple of degrees.
    Returns two arrays:
      mse_tr : np.ndarray shape (len(degrees),)
      mse_te : np.ndarray shape (len(degrees),)
    aligned with degrees.
    """
    # TODO: Implement
    mse_tr = np.array([])
    mse_te = np.array([])
    for degree in degrees:
        train_mse, test_mse = eval_poly(Xtr, ytr, Xte, yte, degree)
        mse_tr = np.append(mse_tr, train_mse)
        mse_te = np.append(mse_te, test_mse)
    return mse_tr, mse_te


def recommend_degree(degrees: Sequence[int], mse_te: np.ndarray) -> int:
    """
    Return the degree with the smallest test MSE.
    Break ties by returning the smaller degree.
    """
    # TODO: Implement
    return degrees[np.argmin(mse_te)]
