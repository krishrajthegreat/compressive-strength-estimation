"""
Gaussian Process regression for concrete compressive strength prediction.

Inspired by Meta's BOxCrete (2026): uses Matérn kernel with log-time
transformation for smooth, continuous strength curves with built-in
uncertainty quantification (posterior variance).

Uses sklearn for lightweight implementation (no PyTorch dependency).
"""

import numpy as np
import pickle
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    Matern, WhiteKernel, ConstantKernel, RBF
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold


def create_gp_model(kernel_type: str = 'matern') -> Pipeline:
    """
    Create a GP regression pipeline with scaling and kernel selection.

    Parameters
    ----------
    kernel_type : str
        Kernel type: 'matern' (default, recommended) or 'rbf'.

    Returns
    -------
    Pipeline
        sklearn Pipeline with StandardScaler + GaussianProcessRegressor.
    """
    if kernel_type == 'matern':
        # Matérn 5/2: smooth but not infinitely differentiable
        # Good for physical processes like hydration
        kernel = (
            ConstantKernel(1.0, constant_value_bounds=(1e-3, 1e3))
            * Matern(nu=2.5, length_scale=1.0, length_scale_bounds=(1e-3, 1e3))
            + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1))
        )
    elif kernel_type == 'rbf':
        kernel = (
            ConstantKernel(1.0, constant_value_bounds=(1e-3, 1e3))
            * RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e3))
            + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1))
        )
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")

    gp = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        n_restarts_optimizer=10,
        random_state=42,
        alpha=1e-6,  # numerical stability
    )

    return Pipeline([
        ('scaler', StandardScaler()),
        ('gp', gp),
    ])


def gp_predict_with_uncertainty(
    model: Pipeline,
    X: np.ndarray,
) -> tuple:
    """
    Generate predictions with uncertainty estimates.

    Parameters
    ----------
    model : Pipeline
        Fitted GP pipeline.
    X : np.ndarray
        Feature matrix.

    Returns
    -------
    tuple
        (mean_predictions, std_predictions)
    """
    # Scale X through the pipeline's scaler
    X_scaled = model.named_steps['scaler'].transform(X)
    mean, std = model.named_steps['gp'].predict(X_scaled, return_std=True)
    return mean, std


def evaluate_gp_cv(
    X, y,
    n_folds: int = 5,
    kernel_type: str = 'matern',
) -> dict:
    """
    Evaluate GP model via k-fold cross-validation.

    Parameters
    ----------
    X : array-like
        Feature matrix.
    y : array-like
        Target vector.
    n_folds : int
        Number of CV folds.
    kernel_type : str
        Kernel type for GP.

    Returns
    -------
    dict
        Metrics including uncertainty calibration.
    """
    X_np = X.values if hasattr(X, 'values') else np.array(X)
    y_np = y.values if hasattr(y, 'values') else np.array(y)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    fold_metrics = {'rmse': [], 'mae': [], 'r2': []}
    coverage_90 = []  # % of true values within 90% prediction interval
    mean_interval_width = []
    y_true_all, y_pred_all, y_std_all = [], [], []

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X_np)):
        X_train, X_test = X_np[train_idx], X_np[test_idx]
        y_train, y_test = y_np[train_idx], y_np[test_idx]

        print(f"    GP fold {fold_idx + 1}/{n_folds}...")

        model = create_gp_model(kernel_type)
        model.fit(X_train, y_train)

        y_pred, y_std = gp_predict_with_uncertainty(model, X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        fold_metrics['rmse'].append(rmse)
        fold_metrics['mae'].append(mae)
        fold_metrics['r2'].append(r2)

        # 90% prediction interval coverage
        z_90 = 1.645
        lower = y_pred - z_90 * y_std
        upper = y_pred + z_90 * y_std
        in_interval = np.sum((y_test >= lower) & (y_test <= upper))
        coverage = in_interval / len(y_test) * 100
        coverage_90.append(coverage)
        mean_interval_width.append(np.mean(2 * z_90 * y_std))

        y_true_all.extend(y_test.tolist())
        y_pred_all.extend(y_pred.tolist())
        y_std_all.extend(y_std.tolist())

        print(f"      RMSE={rmse:.3f}  R²={r2:.4f}  Coverage90={coverage:.1f}%")

    return {
        'RMSE_mean': np.mean(fold_metrics['rmse']),
        'RMSE_std': np.std(fold_metrics['rmse']),
        'MAE_mean': np.mean(fold_metrics['mae']),
        'MAE_std': np.std(fold_metrics['mae']),
        'R2_mean': np.mean(fold_metrics['r2']),
        'R2_std': np.std(fold_metrics['r2']),
        'Coverage90_mean': np.mean(coverage_90),
        'MeanIntervalWidth': np.mean(mean_interval_width),
        'y_true': y_true_all,
        'y_pred': y_pred_all,
        'y_std': y_std_all,
    }


def save_gp_model(model: Pipeline, filepath: str):
    """Save fitted GP model to disk."""
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f"  GP model saved to {filepath}")


def load_gp_model(filepath: str) -> Pipeline:
    """Load a saved GP model."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)
