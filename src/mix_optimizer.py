"""
Bayesian-inspired mix design optimizer.

Uses trained models as surrogates to suggest optimal concrete mix
designs for a target compressive strength, inspired by Meta's BOxCrete.

Uses scipy.optimize (no BoTorch/PyTorch dependency).
"""

import numpy as np
from scipy.optimize import differential_evolution


# Physical bounds for concrete mix components (kg/m³)
DEFAULT_BOUNDS = {
    'Cement': (100, 600),
    'Blast_Furnace_Slag': (0, 400),
    'Fly_Ash': (0, 250),
    'Water': (120, 250),
    'Superplasticizer': (0, 30),
    'Coarse_Aggregate': (700, 1200),
    'Fine_Aggregate': (500, 1000),
}

# Feature column order (must match training)
RAW_FEATURE_ORDER = [
    'Cement', 'Blast_Furnace_Slag', 'Fly_Ash', 'Water',
    'Superplasticizer', 'Coarse_Aggregate', 'Fine_Aggregate', 'Age',
]


def _build_features_from_mix(
    mix: np.ndarray,
    age: float,
    feature_engineer_fn,
) -> np.ndarray:
    """
    Build full feature vector from raw mix components + age.

    Parameters
    ----------
    mix : np.ndarray
        7 raw components: [Cement, Slag, FA, Water, SP, CA, FineA].
    age : float
        Target curing age in days.
    feature_engineer_fn : callable
        Function that takes a DataFrame and returns engineered features.

    Returns
    -------
    np.ndarray
        Full feature vector (1 row).
    """
    import pandas as pd

    row = {
        'Cement': mix[0],
        'Blast_Furnace_Slag': mix[1],
        'Fly_Ash': mix[2],
        'Water': mix[3],
        'Superplasticizer': mix[4],
        'Coarse_Aggregate': mix[5],
        'Fine_Aggregate': mix[6],
        'Age': age,
        'Compressive_Strength': 0.0,  # dummy target
    }
    df = pd.DataFrame([row])
    df_feat = feature_engineer_fn(df)
    feature_cols = [c for c in df_feat.columns if c != 'Compressive_Strength']
    return df_feat[feature_cols].values


def optimize_mix(
    target_strength: float,
    target_age: float,
    model,
    feature_engineer_fn,
    bounds: dict = None,
    wb_range: tuple = (0.25, 0.80),
    n_results: int = 3,
    seed: int = 42,
    maxiter: int = 200,
) -> list:
    """
    Find optimal concrete mix designs for a target strength.

    Parameters
    ----------
    target_strength : float
        Target compressive strength in MPa.
    target_age : float
        Curing age in days.
    model : fitted model
        Trained model with .predict() method.
    feature_engineer_fn : callable
        Feature engineering function.
    bounds : dict, optional
        Component bounds. Uses DEFAULT_BOUNDS if None.
    wb_range : tuple
        Allowed water-to-binder ratio range.
    n_results : int
        Number of top results to return.
    seed : int
        Random seed.
    maxiter : int
        Maximum number of differential evolution iterations.

    Returns
    -------
    list of dict
        Top mix designs with predicted strengths.
    """
    if bounds is None:
        bounds = DEFAULT_BOUNDS

    # Optimization bounds (7 components)
    opt_bounds = [
        bounds['Cement'],
        bounds['Blast_Furnace_Slag'],
        bounds['Fly_Ash'],
        bounds['Water'],
        bounds['Superplasticizer'],
        bounds['Coarse_Aggregate'],
        bounds['Fine_Aggregate'],
    ]

    def objective(x):
        """Minimize squared error to target strength + W/B penalty."""
        cement, slag, fa, water = x[0], x[1], x[2], x[3]
        binder = cement + slag + fa

        # W/B ratio constraint as soft penalty
        if binder < 1e-6:
            return 1e6
        wb = water / binder
        if wb < wb_range[0] or wb > wb_range[1]:
            return 1e6

        try:
            features = _build_features_from_mix(x, target_age, feature_engineer_fn)
            pred = model.predict(features)[0]
            return (pred - target_strength) ** 2
        except Exception:
            return 1e6

    # Run multiple times for diverse results
    results = []
    for i in range(n_results * 3):
        result = differential_evolution(
            objective,
            bounds=opt_bounds,
            seed=seed + i,
            maxiter=maxiter,
            tol=1e-4,
            polish=True,
        )
        if result.fun < 1e5:  # valid solution
            mix = result.x
            features = _build_features_from_mix(mix, target_age, feature_engineer_fn)
            pred = model.predict(features)[0]
            binder = mix[0] + mix[1] + mix[2]

            results.append({
                'Cement': round(mix[0], 1),
                'Slag': round(mix[1], 1),
                'Fly_Ash': round(mix[2], 1),
                'Water': round(mix[3], 1),
                'Superplasticizer': round(mix[4], 2),
                'Coarse_Aggregate': round(mix[5], 1),
                'Fine_Aggregate': round(mix[6], 1),
                'W_B_ratio': round(mix[3] / binder, 3) if binder > 0 else None,
                'Predicted_Strength': round(pred, 2),
                'Error_to_Target': round(abs(pred - target_strength), 2),
                'Total_Binder': round(binder, 1),
            })

    # Sort by closeness to target and return top N
    results.sort(key=lambda r: r['Error_to_Target'])
    return results[:n_results]
