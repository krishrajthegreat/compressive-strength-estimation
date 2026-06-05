"""
Validation module: nested cross-validation, standard CV evaluation,
and group-based (mix-level) cross-validation.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, GroupKFold, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from catboost import CatBoostRegressor
import lightgbm as lgb
import optuna
from optuna.samplers import TPESampler

from src.optimization import OBJECTIVE_MAP


# ── Model factory ───────────────────────────────────────────────────────────

def _create_model(model_type: str, params: dict, monotonic_constraints=None):
    """Instantiate a model with the given hyperparameters and optional constraints."""
    extra = {}
    if monotonic_constraints is not None:
        if model_type == 'XGBoost':
            extra['monotone_constraints'] = tuple(monotonic_constraints)
        elif model_type == 'CatBoost':
            extra['monotone_constraints'] = monotonic_constraints
        elif model_type == 'LightGBM':
            extra['monotone_constraints'] = monotonic_constraints

    if model_type == 'XGBoost':
        return xgb.XGBRegressor(**params, **extra, random_state=42, n_jobs=-1)
    elif model_type == 'CatBoost':
        return CatBoostRegressor(**params, **extra, random_state=42, verbose=0)
    elif model_type == 'LightGBM':
        return lgb.LGBMRegressor(**params, **extra, random_state=42, n_jobs=-1, verbose=-1)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# ── Nested Cross-Validation with Optuna ─────────────────────────────────────

def nested_cv_with_optuna(
    X: pd.DataFrame,
    y: pd.Series,
    model_type: str,
    n_outer_folds: int = 5,
    n_trials: int = 100,
    monotonic_constraints=None,
) -> dict:
    """
    Full nested cross-validation: outer loop for unbiased performance
    estimation, inner Optuna loop for hyperparameter optimization.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target vector.
    model_type : str
        One of 'XGBoost', 'CatBoost', 'LightGBM'.
    n_outer_folds : int
        Number of outer CV folds (default 5).
    n_trials : int
        Optuna trials for inner HPO (default 100).

    Returns
    -------
    dict
        Aggregated metrics (mean ± std) and per-fold predictions/params.
    """
    outer_cv = KFold(n_splits=n_outer_folds, shuffle=True, random_state=42)
    objective_fn = OBJECTIVE_MAP[model_type]

    fold_metrics = {'rmse': [], 'mae': [], 'r2': [], 'mape': [], 'max_error': []}
    y_true_all, y_pred_all = [], []
    best_params_per_fold = []

    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X)):
        print(f"  Outer fold {fold_idx + 1}/{n_outer_folds}")

        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_test = y.iloc[test_idx]

        # Inner loop: Optuna HPO
        study = optuna.create_study(
            direction='minimize', sampler=TPESampler(seed=42 + fold_idx),
        )
        study.optimize(
            lambda trial: objective_fn(trial, X_train, y_train, monotonic_constraints),
            n_trials=n_trials,
            show_progress_bar=False,
        )
        best_params = study.best_params
        best_params_per_fold.append(best_params)

        # Train on outer training set
        model = _create_model(model_type, best_params, monotonic_constraints)
        model.fit(X_train, y_train)

        # Evaluate on outer test set
        y_pred = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # MAPE — guard against zero true values
        mask = y_test != 0
        if mask.sum() > 0:
            mape = np.mean(np.abs((y_test[mask] - y_pred[mask.values]) / y_test[mask])) * 100
        else:
            mape = np.nan

        max_err = np.max(np.abs(y_test.values - y_pred))

        fold_metrics['rmse'].append(rmse)
        fold_metrics['mae'].append(mae)
        fold_metrics['r2'].append(r2)
        fold_metrics['mape'].append(mape)
        fold_metrics['max_error'].append(max_err)

        y_true_all.extend(y_test.values.tolist())
        y_pred_all.extend(y_pred.tolist())

        print(f"    RMSE={rmse:.3f}  MAE={mae:.3f}  R²={r2:.4f}")

    return {
        'RMSE_mean': np.mean(fold_metrics['rmse']),
        'RMSE_std': np.std(fold_metrics['rmse']),
        'MAE_mean': np.mean(fold_metrics['mae']),
        'MAE_std': np.std(fold_metrics['mae']),
        'R2_mean': np.mean(fold_metrics['r2']),
        'R2_std': np.std(fold_metrics['r2']),
        'MAPE_mean': np.nanmean(fold_metrics['mape']),
        'MAPE_std': np.nanstd(fold_metrics['mape']),
        'MaxError_mean': np.mean(fold_metrics['max_error']),
        'MaxError_std': np.std(fold_metrics['max_error']),
        'y_true': y_true_all,
        'y_pred': y_pred_all,
        'best_params_per_fold': best_params_per_fold,
    }


# ── Standard CV evaluation (with fixed best params) ────────────────────────

def cv_evaluation(
    X: pd.DataFrame,
    y: pd.Series,
    model_type: str,
    best_params: dict,
    n_folds: int = 5,
) -> dict:
    """
    Evaluate model with fixed hyperparameters across k-fold CV.

    This is used for final reporting *after* nested CV has provided
    unbiased estimates and a representative param set has been selected.
    """
    outer_cv = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    fold_metrics = {'rmse': [], 'mae': [], 'r2': []}
    y_true_all, y_pred_all = [], []

    for train_idx, test_idx in outer_cv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = _create_model(model_type, best_params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        fold_metrics['rmse'].append(np.sqrt(mean_squared_error(y_test, y_pred)))
        fold_metrics['mae'].append(mean_absolute_error(y_test, y_pred))
        fold_metrics['r2'].append(r2_score(y_test, y_pred))

        y_true_all.extend(y_test.values.tolist())
        y_pred_all.extend(y_pred.tolist())

    return {
        'RMSE_mean': np.mean(fold_metrics['rmse']),
        'RMSE_std': np.std(fold_metrics['rmse']),
        'MAE_mean': np.mean(fold_metrics['mae']),
        'MAE_std': np.std(fold_metrics['mae']),
        'R2_mean': np.mean(fold_metrics['r2']),
        'R2_std': np.std(fold_metrics['r2']),
        'y_true': y_true_all,
        'y_pred': y_pred_all,
    }


# ── Group-based CV (mix-level generalization) ───────────────────────────────

def group_cv_evaluation(
    X: pd.DataFrame,
    y: pd.Series,
    model_type: str,
    best_params: dict,
    df_raw: pd.DataFrame,
    n_folds: int = 5,
) -> dict:
    """
    GroupKFold ensuring all ages of a mix design are either in train or test.

    Parameters
    ----------
    df_raw : pd.DataFrame
        Original raw DataFrame (before feature engineering) to derive mix IDs.
    """
    mix_cols = [
        'Cement', 'Blast_Furnace_Slag', 'Fly_Ash', 'Water',
        'Superplasticizer', 'Coarse_Aggregate', 'Fine_Aggregate',
    ]
    mix_ids = df_raw[mix_cols].apply(
        lambda row: '_'.join(row.astype(str)), axis=1
    )

    n_unique_mixes = mix_ids.nunique()
    actual_folds = min(n_folds, n_unique_mixes)
    if actual_folds < n_folds:
        print(f"  WARNING: Only {n_unique_mixes} unique mixes; "
              f"using {actual_folds} folds instead of {n_folds}.")

    group_cv = GroupKFold(n_splits=actual_folds)

    fold_metrics = {'rmse': [], 'mae': [], 'r2': []}
    y_true_all, y_pred_all = [], []

    for train_idx, test_idx in group_cv.split(X, y, groups=mix_ids):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = _create_model(model_type, best_params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        fold_metrics['rmse'].append(np.sqrt(mean_squared_error(y_test, y_pred)))
        fold_metrics['mae'].append(mean_absolute_error(y_test, y_pred))
        fold_metrics['r2'].append(r2_score(y_test, y_pred))

        y_true_all.extend(y_test.values.tolist())
        y_pred_all.extend(y_pred.tolist())

    return {
        'RMSE_mean': np.mean(fold_metrics['rmse']),
        'RMSE_std': np.std(fold_metrics['rmse']),
        'MAE_mean': np.mean(fold_metrics['mae']),
        'MAE_std': np.std(fold_metrics['mae']),
        'R2_mean': np.mean(fold_metrics['r2']),
        'R2_std': np.std(fold_metrics['r2']),
        'y_true': y_true_all,
        'y_pred': y_pred_all,
    }


# ── Baseline models ────────────────────────────────────────────────────────

def evaluate_baselines(X: pd.DataFrame, y: pd.Series, n_folds: int = 5) -> dict:
    """
    Evaluate Linear Regression and Random Forest baselines via k-fold CV.

    Returns
    -------
    dict
        Keys = model names, values = metric dictionaries.
    """
    baselines = {
        'LinearRegression': LinearRegression(),
        'RandomForest': RandomForestRegressor(
            n_estimators=200, max_depth=10, random_state=42, n_jobs=-1
        ),
    }

    results = {}
    cv = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    for name, model in baselines.items():
        fold_metrics = {'rmse': [], 'mae': [], 'r2': []}
        y_true_all, y_pred_all = [], []

        for train_idx, test_idx in cv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            model_clone = type(model)(**model.get_params())
            model_clone.fit(X_train, y_train)
            y_pred = model_clone.predict(X_test)

            fold_metrics['rmse'].append(np.sqrt(mean_squared_error(y_test, y_pred)))
            fold_metrics['mae'].append(mean_absolute_error(y_test, y_pred))
            fold_metrics['r2'].append(r2_score(y_test, y_pred))

            y_true_all.extend(y_test.values.tolist())
            y_pred_all.extend(y_pred.tolist())

        results[name] = {
            'RMSE_mean': np.mean(fold_metrics['rmse']),
            'RMSE_std': np.std(fold_metrics['rmse']),
            'MAE_mean': np.mean(fold_metrics['mae']),
            'MAE_std': np.std(fold_metrics['mae']),
            'R2_mean': np.mean(fold_metrics['r2']),
            'R2_std': np.std(fold_metrics['r2']),
            'y_true': y_true_all,
            'y_pred': y_pred_all,
        }

        print(f"  {name}: RMSE={results[name]['RMSE_mean']:.3f}±{results[name]['RMSE_std']:.3f}, "
              f"R²={results[name]['R2_mean']:.3f}±{results[name]['R2_std']:.3f}")

    return results
