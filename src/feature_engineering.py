"""
Feature engineering based on concrete engineering domain knowledge.

Stage A features (14): Binder system, aggregate, admixture, temporal transforms.
Stage B additions (8): Physics-grounded ratios, interaction terms, gel-space ratio.

Total: 8 raw + 22 engineered = 30 features.
"""

import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def engineer_features(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """
    Create derived features from raw mix design and curing parameters.

    Stage A (14 features):
      Binder system (5): Binder, W_B_ratio, GGBS_ratio, FlyAsh_ratio, SCM_ratio
      Aggregate (3):     Total_Aggregate, Fine_Agg_ratio, Agg_Binder_ratio
      Admixture (1):     SP_per_binder
      Temporal (5):      log_Age, sqrt_Age, Age_very_early, Age_early, Age_standard

    Stage B (8 features):
      Physics (4):       W_C_ratio, Cement_fraction, gel_space_ratio, effective_WB
      Interactions (4):  age_wb_interaction, GGBS_age_interaction,
                         FlyAsh_age_interaction, Binder_intensity

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with the 8 raw input features.

    Returns
    -------
    pd.DataFrame
        DataFrame with original + derived features (30 total).
    """
    df_feat = df.copy()

    # ── Binder system features (Stage A) ─────────────────────────────────
    # Total binder = cement + supplementary cementitious materials
    df_feat['Binder'] = (
        df_feat['Cement']
        + df_feat['Blast_Furnace_Slag']
        + df_feat['Fly_Ash']
    )

    # Water-to-binder ratio — single most important predictor (ACI 211)
    df_feat['W_B_ratio'] = df_feat['Water'] / df_feat['Binder']

    # SCM replacement ratios
    df_feat['GGBS_ratio'] = df_feat['Blast_Furnace_Slag'] / df_feat['Binder']
    df_feat['FlyAsh_ratio'] = df_feat['Fly_Ash'] / df_feat['Binder']
    df_feat['SCM_ratio'] = (
        (df_feat['Blast_Furnace_Slag'] + df_feat['Fly_Ash'])
        / df_feat['Binder']
    )

    # ── Aggregate features (Stage A) ─────────────────────────────────────
    df_feat['Total_Aggregate'] = (
        df_feat['Coarse_Aggregate'] + df_feat['Fine_Aggregate']
    )
    df_feat['Fine_Agg_ratio'] = (
        df_feat['Fine_Aggregate'] / df_feat['Total_Aggregate']
    )
    df_feat['Agg_Binder_ratio'] = (
        df_feat['Total_Aggregate'] / df_feat['Binder']
    )

    # ── Admixture intensity (Stage A) ────────────────────────────────────
    df_feat['SP_per_binder'] = df_feat['Superplasticizer'] / df_feat['Binder']

    # ── Temporal transformations (Stage A) ───────────────────────────────
    # Log-transform: strength develops log-linearly after ~7 days
    df_feat['log_Age'] = np.log1p(df_feat['Age'])

    # Square root: intermediate growth model
    df_feat['sqrt_Age'] = np.sqrt(df_feat['Age'])

    # Hydration phase indicators
    df_feat['Age_very_early'] = (df_feat['Age'] <= 3).astype(int)
    df_feat['Age_early'] = (
        (df_feat['Age'] > 3) & (df_feat['Age'] <= 7)
    ).astype(int)
    df_feat['Age_standard'] = (
        (df_feat['Age'] > 7) & (df_feat['Age'] <= 28)
    ).astype(int)

    # ── Stage B: Physics-grounded features ───────────────────────────────

    # Water-to-cement ratio (classic Abrams' law, distinct from W/B)
    df_feat['W_C_ratio'] = df_feat['Water'] / df_feat['Cement'].clip(lower=1e-6)

    # Cement fraction of total binder (OPC dominance indicator)
    df_feat['Cement_fraction'] = df_feat['Cement'] / df_feat['Binder']

    # Gel-space ratio (Powers' theory) — most fundamental strength predictor
    # α = degree of hydration, approximated by: 1 - exp(-0.4 * Age^0.5)
    # gel_space = 0.68 * α * C / (0.68 * α * C + W)
    alpha = 1.0 - np.exp(-0.4 * np.sqrt(df_feat['Age']))
    gel_numerator = 0.68 * alpha * df_feat['Cement']
    df_feat['gel_space_ratio'] = (
        gel_numerator / (gel_numerator + df_feat['Water']).clip(lower=1e-6)
    )

    # Effective W/B ratio — age-dependent pozzolanic contribution
    # At early ages, SCMs contribute less; k(t) increases with age
    # k(t) approximation: 0.3 for age<=3, 0.6 for age<=14, 0.9 for age>14
    k_t = np.where(
        df_feat['Age'] <= 3, 0.3,
        np.where(df_feat['Age'] <= 14, 0.6, 0.9)
    )
    effective_binder = (
        df_feat['Cement']
        + k_t * (df_feat['Blast_Furnace_Slag'] + df_feat['Fly_Ash'])
    )
    df_feat['effective_WB'] = df_feat['Water'] / effective_binder.clip(lower=1e-6)

    # ── Stage B: Interaction terms ───────────────────────────────────────

    # Age × W/B interaction — captures age-dependent W/B sensitivity
    df_feat['age_wb_interaction'] = df_feat['log_Age'] * df_feat['W_B_ratio']

    # GGBS activation over time (slag reacts slowly)
    df_feat['GGBS_age_interaction'] = (
        df_feat['GGBS_ratio'] * df_feat['log_Age']
    )

    # Fly ash pozzolanic delay (FA even slower than slag)
    df_feat['FlyAsh_age_interaction'] = (
        df_feat['FlyAsh_ratio'] * df_feat['log_Age']
    )

    # Weighted effective binder intensity
    # Early-age activation factors: Slag=0.4, FA=0.2 (from augmentation exp.)
    df_feat['Binder_intensity'] = (
        (df_feat['Cement'] + 0.4 * df_feat['Blast_Furnace_Slag']
         + 0.2 * df_feat['Fly_Ash'])
        / df_feat['W_C_ratio'].clip(lower=1e-6)
    )

    # ── Sanity checks ───────────────────────────────────────────────────
    # Replace any infinities from division by zero
    df_feat.replace([np.inf, -np.inf], np.nan, inplace=True)

    n_nan = df_feat.isnull().sum().sum()
    if n_nan > 0:
        nan_cols = df_feat.columns[df_feat.isnull().any()].tolist()
        logger.warning(
            f"{n_nan} NaN values created during feature engineering "
            f"in columns: {nan_cols}. Filling with 0."
        )
        df_feat.fillna(0, inplace=True)

    if verbose:
        n_raw = 8
        n_engineered = len(df_feat.columns) - len(df.columns)
        n_total = len(df_feat.columns) - 1  # exclude target
        print(f"\nFeature engineering complete.")
        print(f"  Raw features:       {n_raw}")
        print(f"  Engineered features: {n_engineered}")
        print(f"  Total features:     {n_total} (excl. target)")

    return df_feat


def get_feature_columns(df: pd.DataFrame) -> list:
    """
    Return all feature column names (excluding the target).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with all features and target.

    Returns
    -------
    list
        Feature column names.
    """
    return [c for c in df.columns if c != 'Compressive_Strength']
