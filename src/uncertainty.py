"""
Uncertainty quantification via Conformal Prediction.

Provides model-agnostic prediction intervals with statistical coverage
guarantees. Works with any point-prediction model (trees, GP, stacking).

Methods:
  - Split Conformal Prediction: fixed-width intervals
  - Conformalized Quantile Regression (CQR): adaptive intervals
"""

import numpy as np
from sklearn.model_selection import KFold


class SplitConformalPredictor:
    """
    Split conformal prediction for regression.

    Produces prediction intervals [ŷ - q, ŷ + q] where q is the
    (1-α) quantile of calibration residuals.

    Parameters
    ----------
    alpha : float
        Miscoverage rate. E.g., alpha=0.10 → 90% coverage target.
    """

    def __init__(self, alpha: float = 0.10):
        self.alpha = alpha
        self.q_ = None
        self.scores_ = None

    def calibrate(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Calibrate on held-out data.

        Parameters
        ----------
        y_true : np.ndarray
            True target values (calibration set).
        y_pred : np.ndarray
            Model predictions on calibration set.
        """
        self.scores_ = np.abs(y_true - y_pred)
        n = len(self.scores_)
        # Finite-sample correction: ceil((n+1)(1-α)) / n
        level = np.ceil((n + 1) * (1 - self.alpha)) / n
        level = min(level, 1.0)
        self.q_ = np.quantile(self.scores_, level)

    def predict_interval(
        self, y_pred: np.ndarray
    ) -> tuple:
        """
        Generate prediction intervals.

        Parameters
        ----------
        y_pred : np.ndarray
            Point predictions on new data.

        Returns
        -------
        tuple
            (lower_bounds, upper_bounds)
        """
        if self.q_ is None:
            raise RuntimeError("Must call calibrate() first.")
        return y_pred - self.q_, y_pred + self.q_

    def get_interval_width(self) -> float:
        """Return the fixed interval half-width."""
        return self.q_ if self.q_ is not None else 0.0


def evaluate_conformal_cv(
    model_class_or_factory,
    X: np.ndarray,
    y: np.ndarray,
    alpha: float = 0.10,
    n_folds: int = 5,
    fit_kwargs: dict = None,
) -> dict:
    """
    Evaluate conformal prediction via cross-validation.

    For each fold:
      1. Split train into fit + calibration (80/20)
      2. Fit model on fit set
      3. Calibrate conformal on calibration set
      4. Evaluate intervals on test set

    Parameters
    ----------
    model_class_or_factory : callable
        Function that returns an unfitted model instance.
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Target vector.
    alpha : float
        Miscoverage rate.
    n_folds : int
        Number of CV folds.
    fit_kwargs : dict, optional
        Extra kwargs for model.fit().

    Returns
    -------
    dict
        Coverage, interval width, and point prediction metrics.
    """
    if fit_kwargs is None:
        fit_kwargs = {}

    X_np = X.values if hasattr(X, 'values') else np.array(X)
    y_np = y.values if hasattr(y, 'values') else np.array(y)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    coverages = []
    widths = []
    rmses = []

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X_np)):
        X_train_full, X_test = X_np[train_idx], X_np[test_idx]
        y_train_full, y_test = y_np[train_idx], y_np[test_idx]

        # Split train into fit + calibration (80/20)
        n_train = len(X_train_full)
        n_cal = max(int(n_train * 0.2), 10)
        cal_idx = np.random.RandomState(42 + fold_idx).choice(
            n_train, size=n_cal, replace=False
        )
        fit_idx = np.setdiff1d(np.arange(n_train), cal_idx)

        X_fit, X_cal = X_train_full[fit_idx], X_train_full[cal_idx]
        y_fit, y_cal = y_train_full[fit_idx], y_train_full[cal_idx]

        # Fit model
        model = model_class_or_factory()
        model.fit(X_fit, y_fit, **fit_kwargs)

        # Calibrate conformal
        cp = SplitConformalPredictor(alpha=alpha)
        y_cal_pred = model.predict(X_cal)
        cp.calibrate(y_cal, y_cal_pred)

        # Evaluate
        y_test_pred = model.predict(X_test)
        lower, upper = cp.predict_interval(y_test_pred)

        # Metrics
        in_interval = np.sum((y_test >= lower) & (y_test <= upper))
        coverage = in_interval / len(y_test) * 100
        width = cp.get_interval_width() * 2

        from sklearn.metrics import mean_squared_error
        rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

        coverages.append(coverage)
        widths.append(width)
        rmses.append(rmse)

    return {
        'coverage_mean': np.mean(coverages),
        'coverage_std': np.std(coverages),
        'interval_width_mean': np.mean(widths),
        'interval_width_std': np.std(widths),
        'rmse_mean': np.mean(rmses),
        'target_coverage': (1 - alpha) * 100,
    }
