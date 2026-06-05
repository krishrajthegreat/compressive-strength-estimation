"""
Stacking ensemble for concrete compressive strength prediction.

Architecture:
    Base learners: CatBoost, XGBoost, LightGBM (with monotonic constraints)
    Meta-learner:  Ridge regression on out-of-fold (OOF) predictions

Key principle: OOF predictions prevent data leakage in the meta-learner.
"""

import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
from catboost import CatBoostRegressor
import lightgbm as lgb


def create_base_model(model_type: str, params: dict, monotonic_constraints=None):
    """
    Instantiate a base model with hyperparameters and optional constraints.

    Parameters
    ----------
    model_type : str
        One of 'XGBoost', 'CatBoost', 'LightGBM'.
    params : dict
        Best hyperparameters from Optuna.
    monotonic_constraints : list or dict, optional
        Monotonic constraint specification.

    Returns
    -------
    model
        Instantiated (unfitted) model.
    """
    if model_type == 'XGBoost':
        extra = {}
        if monotonic_constraints is not None:
            extra['monotone_constraints'] = tuple(monotonic_constraints)
        return xgb.XGBRegressor(**params, **extra, random_state=42, n_jobs=-1)

    elif model_type == 'CatBoost':
        extra = {}
        if monotonic_constraints is not None:
            extra['monotone_constraints'] = monotonic_constraints
        return CatBoostRegressor(**params, **extra, random_state=42, verbose=0)

    elif model_type == 'LightGBM':
        extra = {}
        if monotonic_constraints is not None:
            extra['monotone_constraints'] = monotonic_constraints
        return lgb.LGBMRegressor(**params, **extra, random_state=42, n_jobs=-1, verbose=-1)

    else:
        raise ValueError(f"Unknown model type: {model_type}")


def generate_oof_predictions(
    X: np.ndarray,
    y: np.ndarray,
    model_type: str,
    params: dict,
    n_splits: int = 5,
    monotonic_constraints=None,
) -> np.ndarray:
    """
    Generate out-of-fold predictions for stacking.

    For each fold, the model is trained on K-1 folds and predicts the
    held-out fold. This ensures every sample has a prediction made when
    it was NOT in the training set — preventing meta-learner leakage.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Target vector.
    model_type : str
        Model identifier.
    params : dict
        Best hyperparameters.
    n_splits : int
        Number of CV folds for OOF generation.
    monotonic_constraints : list or dict, optional
        Monotonic constraint specification.

    Returns
    -------
    np.ndarray
        OOF predictions aligned with original indices.
    """
    oof_preds = np.zeros(len(X))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train = y[train_idx]

        model = create_base_model(model_type, params, monotonic_constraints)
        model.fit(X_train, y_train)
        oof_preds[val_idx] = model.predict(X_val)

    return oof_preds


class StackingEnsemble:
    """
    Stacking ensemble with OOF-based meta-learner training.

    Parameters
    ----------
    base_configs : dict
        Mapping of model_type -> {'params': dict, 'constraints': list/dict}.
    meta_alpha : float
        Ridge regression regularization strength.
    n_oof_splits : int
        Number of CV folds for OOF generation.
    """

    def __init__(self, base_configs: dict, meta_alpha: float = 1.0,
                 n_oof_splits: int = 5):
        self.base_configs = base_configs
        self.meta_learner = Ridge(alpha=meta_alpha)
        self.n_oof_splits = n_oof_splits
        self.base_models_ = {}  # Fitted base models
        self.meta_weights_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the stacking ensemble:
          1. Generate OOF predictions for each base model
          2. Train meta-learner on OOF predictions
          3. Refit base models on full data for inference
        """
        # Step 1: Generate OOF predictions
        oof_matrix = []
        for model_type, config in self.base_configs.items():
            print(f"    Generating OOF for {model_type}...")
            oof = generate_oof_predictions(
                X, y,
                model_type=model_type,
                params=config['params'],
                n_splits=self.n_oof_splits,
                monotonic_constraints=config.get('constraints'),
            )
            oof_matrix.append(oof)

        oof_matrix = np.column_stack(oof_matrix)

        # Step 2: Train meta-learner on OOF predictions
        self.meta_learner.fit(oof_matrix, y)
        self.meta_weights_ = {
            name: coef for name, coef in
            zip(self.base_configs.keys(), self.meta_learner.coef_)
        }
        print(f"    Meta-learner weights: {self.meta_weights_}")
        print(f"    Meta-learner intercept: {self.meta_learner.intercept_:.4f}")

        # Step 3: Refit base models on full training data
        for model_type, config in self.base_configs.items():
            model = create_base_model(
                model_type, config['params'], config.get('constraints')
            )
            model.fit(X, y)
            self.base_models_[model_type] = model

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate stacked predictions."""
        base_preds = np.column_stack([
            model.predict(X) for model in self.base_models_.values()
        ])
        return self.meta_learner.predict(base_preds)

    def predict_base(self, X: np.ndarray) -> dict:
        """Get individual base model predictions (for diagnostics)."""
        return {
            name: model.predict(X)
            for name, model in self.base_models_.items()
        }

    def save(self, filepath: str):
        """Save the entire stacking ensemble to disk."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'base_models': self.base_models_,
                'meta_learner': self.meta_learner,
                'base_configs': self.base_configs,
                'meta_weights': self.meta_weights_,
            }, f)
        print(f"  Stacking ensemble saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'StackingEnsemble':
        """Load a saved stacking ensemble."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        ensemble = cls(base_configs=data['base_configs'])
        ensemble.base_models_ = data['base_models']
        ensemble.meta_learner = data['meta_learner']
        ensemble.meta_weights_ = data['meta_weights']
        return ensemble


def evaluate_stacking_nested_cv(
    X: pd.DataFrame,
    y: pd.Series,
    base_configs: dict,
    n_outer_folds: int = 5,
    n_inner_trials: int = 100,
    meta_alpha: float = 1.0,
) -> dict:
    """
    Evaluate stacking ensemble via nested cross-validation.

    Outer loop: evaluation folds.
    Inner loop: For each outer training set:
      1. Run Optuna HPO for each base model
      2. Generate OOF predictions
      3. Train meta-learner
      4. Evaluate on outer test set

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target vector.
    base_configs : dict
        Mapping of model_type -> {'constraints': list/dict}.
        Params will be optimized per fold.
    n_outer_folds : int
        Outer CV folds.
    n_inner_trials : int
        Optuna trials per model in inner HPO.
    meta_alpha : float
        Ridge regularization.

    Returns
    -------
    dict
        Aggregated metrics and per-fold details.
    """
    from src.optimization import OBJECTIVE_MAP

    outer_cv = KFold(n_splits=n_outer_folds, shuffle=True, random_state=42)
    fold_metrics = {'rmse': [], 'mae': [], 'r2': [], 'mape': []}
    y_true_all, y_pred_all = [], []

    X_np = X.values if hasattr(X, 'values') else np.array(X)
    y_np = y.values if hasattr(y, 'values') else np.array(y)

    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X_np)):
        print(f"\n  Stacking outer fold {fold_idx + 1}/{n_outer_folds}")
        X_train, X_test = X_np[train_idx], X_np[test_idx]
        y_train, y_test = y_np[train_idx], y_np[test_idx]

        # Convert to DataFrame/Series for Optuna compatibility
        X_train_df = pd.DataFrame(X_train, columns=X.columns)
        y_train_s = pd.Series(y_train)

        # Step 1: Optuna HPO for each base model
        fold_configs = {}
        import optuna
        from optuna.samplers import TPESampler

        for model_type, config in base_configs.items():
            print(f"    HPO for {model_type}...")
            objective_fn = OBJECTIVE_MAP[model_type]
            constraints = config.get('constraints')

            study = optuna.create_study(
                direction='minimize',
                sampler=TPESampler(seed=42 + fold_idx),
            )
            study.optimize(
                lambda trial, mt=model_type, c=constraints: (
                    OBJECTIVE_MAP[mt](trial, X_train_df, y_train_s, c)
                ),
                n_trials=n_inner_trials,
                show_progress_bar=False,
            )
            fold_configs[model_type] = {
                'params': study.best_params,
                'constraints': constraints,
            }
            print(f"      Best RMSE: {study.best_value:.3f}")

        # Step 2: Build and fit stacking ensemble
        stack = StackingEnsemble(
            base_configs=fold_configs,
            meta_alpha=meta_alpha,
        )
        stack.fit(X_train, y_train)

        # Step 3: Evaluate on outer test set
        y_pred = stack.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        mask = y_test != 0
        mape = np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100

        fold_metrics['rmse'].append(rmse)
        fold_metrics['mae'].append(mae)
        fold_metrics['r2'].append(r2)
        fold_metrics['mape'].append(mape)

        y_true_all.extend(y_test.tolist())
        y_pred_all.extend(y_pred.tolist())

        print(f"    Fold {fold_idx+1}: RMSE={rmse:.3f}  MAE={mae:.3f}  R²={r2:.4f}")

    return {
        'RMSE_mean': np.mean(fold_metrics['rmse']),
        'RMSE_std': np.std(fold_metrics['rmse']),
        'MAE_mean': np.mean(fold_metrics['mae']),
        'MAE_std': np.std(fold_metrics['mae']),
        'R2_mean': np.mean(fold_metrics['r2']),
        'R2_std': np.std(fold_metrics['r2']),
        'MAPE_mean': np.nanmean(fold_metrics['mape']),
        'y_true': y_true_all,
        'y_pred': y_pred_all,
    }
