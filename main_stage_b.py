#!/usr/bin/env python3
"""
Stage B: Physics-Guided Stacking Ensemble Pipeline

Main orchestrator for Stage B improvements:
  1. Data loading & preprocessing (same as Stage A)
  2. Enhanced feature engineering (22 → 30 features)
  3. Monotonic constraints on all GBDT models
  4. Individual model evaluation via nested CV
  5. Stacking ensemble (CatBoost + XGBoost + LightGBM → Ridge)
  6. Gaussian Process regression with uncertainty
  7. Conformal prediction intervals
  8. Results comparison (Stage A vs Stage B)

Usage:
    python main_stage_b.py                     # Full run (~6-10 hours)
    python main_stage_b.py --quick             # Smoke test (~15-30 min)
    python main_stage_b.py --n-trials 50       # Custom trials
    python main_stage_b.py --skip-gp           # Skip GP (saves time)
    python main_stage_b.py --skip-stacking     # Skip stacking
    python main_stage_b.py --ablation          # Run full ablation study
"""

import argparse
import json
import os
import sys
import time
import pickle
import warnings
import logging

import numpy as np
import pandas as pd
import optuna

# Suppress verbose warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import load_and_preprocess_data, create_age_subsets
from src.feature_engineering import engineer_features, get_feature_columns
from src.optimization import (
    run_optimization,
    build_monotonic_constraints,
    build_catboost_constraints,
    OBJECTIVE_MAP,
)
from src.validation import (
    nested_cv_with_optuna,
    evaluate_baselines,
)
from src.ensemble import (
    StackingEnsemble,
    evaluate_stacking_nested_cv,
    create_base_model,
)
from src.gp_model import (
    create_gp_model,
    evaluate_gp_cv,
    save_gp_model,
)
from src.uncertainty import (
    SplitConformalPredictor,
    evaluate_conformal_cv,
)

np.random.seed(42)

MODEL_TYPES = ['XGBoost', 'CatBoost', 'LightGBM']


def parse_args():
    parser = argparse.ArgumentParser(
        description='Stage B: Physics-Guided Stacking Ensemble Pipeline',
    )
    parser.add_argument(
        '--data', type=str, default='data/Concrete_Data - Sheet1.csv',
        help='Path to CSV dataset',
    )
    parser.add_argument(
        '--n-trials', type=int, default=100,
        help='Optuna trials per model-subset (default: 100)',
    )
    parser.add_argument(
        '--quick', action='store_true',
        help='Quick mode: 10 trials, Full subset only, skip GP',
    )
    parser.add_argument(
        '--output-dir', type=str, default='outputs_stage_b',
        help='Output directory (default: outputs_stage_b)',
    )
    parser.add_argument(
        '--model-dir', type=str, default='models_stage_b',
        help='Directory for saved model weights (default: models_stage_b)',
    )
    parser.add_argument(
        '--skip-gp', action='store_true',
        help='Skip Gaussian Process evaluation',
    )
    parser.add_argument(
        '--skip-stacking', action='store_true',
        help='Skip stacking ensemble evaluation',
    )
    parser.add_argument(
        '--skip-conformal', action='store_true',
        help='Skip conformal prediction evaluation',
    )
    parser.add_argument(
        '--ablation', action='store_true',
        help='Run full ablation study (A→F experiments)',
    )
    return parser.parse_args()


def _create_final_model(model_type: str, best_params: dict,
                        monotonic_constraints=None):
    """Instantiate a model for final training / saving."""
    return create_base_model(model_type, best_params, monotonic_constraints)


def main():
    args = parse_args()

    n_trials = 10 if args.quick else args.n_trials
    output_dir = args.output_dir
    model_dir = args.model_dir
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    print("=" * 80)
    print("  STAGE B: PHYSICS-GUIDED STACKING ENSEMBLE PIPELINE")
    print("=" * 80)
    print(f"  Trials per model-subset: {n_trials}")
    print(f"  Quick mode:              {args.quick}")
    print(f"  Output directory:        {output_dir}")
    print(f"  Model directory:         {model_dir}")
    print(f"  Skip GP:                 {args.skip_gp or args.quick}")
    print(f"  Skip Stacking:           {args.skip_stacking}")
    print(f"  Skip Conformal:          {args.skip_conformal}")
    print("=" * 80)

    pipeline_start = time.time()

    # ── STEP 1: Data loading ────────────────────────────────────────────
    print("\n[1/8] Loading and preprocessing data...")
    if not os.path.exists(args.data):
        print(f"\nERROR: Dataset not found at '{args.data}'.")
        sys.exit(1)

    df = load_and_preprocess_data(args.data)

    # ── STEP 2: Feature engineering (Stage B: 30 features) ──────────────
    print("\n[2/8] Engineering features (Stage B: 30 features)...")
    df_feat = engineer_features(df, verbose=True)
    feature_cols = get_feature_columns(df_feat)
    print(f"  Features: {feature_cols}")

    # ── STEP 3: Create age subsets ──────────────────────────────────────
    print("\n[3/8] Creating early-age subsets...")
    all_subsets = create_age_subsets(df_feat)

    if args.quick:
        subsets = {'Full': all_subsets['Full']}
        print("  [QUICK MODE] Only evaluating Full subset.")
    else:
        subsets = all_subsets

    # ── STEP 4: Build monotonic constraints ─────────────────────────────
    print("\n[4/8] Building monotonic constraints...")
    mc_list = build_monotonic_constraints(feature_cols)
    mc_catboost = build_catboost_constraints(feature_cols)
    constrained_features = [
        f"{col} ({'+1' if mc_list[i] > 0 else '-1'})"
        for i, col in enumerate(feature_cols)
        if mc_list[i] != 0
    ]
    print(f"  Constrained features: {constrained_features}")

    # ── STEP 5: Individual model evaluation (with constraints) ──────────
    print("\n[5/8] Training constrained gradient boosting models...")
    all_results = []
    all_best_params = {}

    for subset_name, subset_df in subsets.items():
        print(f"\n{'=' * 70}")
        print(f"  Subset: {subset_name} ({len(subset_df)} samples)")
        print(f"{'=' * 70}")

        X = subset_df[feature_cols]
        y = subset_df['Compressive_Strength']

        # Baselines
        print(f"\n  Baselines:")
        baseline_results = evaluate_baselines(X, y)
        for bname, bmetrics in baseline_results.items():
            all_results.append({
                'Subset': subset_name, 'Model': bname,
                'RMSE_mean': bmetrics['RMSE_mean'],
                'RMSE_std': bmetrics['RMSE_std'],
                'MAE_mean': bmetrics['MAE_mean'],
                'MAE_std': bmetrics['MAE_std'],
                'R2_mean': bmetrics['R2_mean'],
                'R2_std': bmetrics['R2_std'],
            })

        for model_type in MODEL_TYPES:
            print(f"\n  -- {model_type} (with monotonic constraints) --")
            model_start = time.time()

            # Get constraints in the right format
            if model_type == 'CatBoost':
                constraints = mc_catboost
            else:
                constraints = mc_list

            # Nested CV with constraints
            results = nested_cv_with_optuna(
                X, y, model_type,
                n_outer_folds=5, n_trials=n_trials,
                monotonic_constraints=constraints,
            )

            elapsed = time.time() - model_start
            print(f"\n  Results ({elapsed:.0f}s):")
            print(f"    RMSE: {results['RMSE_mean']:.3f} ± {results['RMSE_std']:.3f}")
            print(f"    MAE:  {results['MAE_mean']:.3f} ± {results['MAE_std']:.3f}")
            print(f"    R²:   {results['R2_mean']:.4f} ± {results['R2_std']:.4f}")

            all_results.append({
                'Subset': subset_name, 'Model': f'{model_type}_constrained',
                'RMSE_mean': results['RMSE_mean'],
                'RMSE_std': results['RMSE_std'],
                'MAE_mean': results['MAE_mean'],
                'MAE_std': results['MAE_std'],
                'R2_mean': results['R2_mean'],
                'R2_std': results['R2_std'],
            })

            best_params = results['best_params_per_fold'][0]
            all_best_params[f"{model_type}_{subset_name}"] = best_params

            # Save individual model
            print(f"  Saving {model_type} model for {subset_name}...")
            final_model = _create_final_model(model_type, best_params, constraints)
            final_model.fit(X, y)
            model_path = os.path.join(model_dir, f"{model_type}_{subset_name}.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(final_model, f)

    # ── STEP 6: Stacking ensemble ───────────────────────────────────────
    if not args.skip_stacking:
        print("\n[6/8] Evaluating stacking ensemble...")

        for subset_name, subset_df in subsets.items():
            print(f"\n  Stacking for {subset_name}...")
            X = subset_df[feature_cols]
            y = subset_df['Compressive_Strength']

            base_configs = {
                'XGBoost': {'constraints': mc_list},
                'CatBoost': {'constraints': mc_catboost},
                'LightGBM': {'constraints': mc_list},
            }

            stack_results = evaluate_stacking_nested_cv(
                X, y,
                base_configs=base_configs,
                n_outer_folds=5,
                n_inner_trials=n_trials,
                meta_alpha=1.0,
            )

            print(f"\n  Stacking {subset_name} Results:")
            print(f"    RMSE: {stack_results['RMSE_mean']:.3f} ± {stack_results['RMSE_std']:.3f}")
            print(f"    R²:   {stack_results['R2_mean']:.4f} ± {stack_results['R2_std']:.4f}")

            all_results.append({
                'Subset': subset_name, 'Model': 'Stacking_Ensemble',
                'RMSE_mean': stack_results['RMSE_mean'],
                'RMSE_std': stack_results['RMSE_std'],
                'MAE_mean': stack_results['MAE_mean'],
                'MAE_std': stack_results['MAE_std'],
                'R2_mean': stack_results['R2_mean'],
                'R2_std': stack_results['R2_std'],
            })

            # Train and save final stacking model on full data
            print(f"  Training final stacking model for {subset_name}...")
            final_configs = {}
            for mt in MODEL_TYPES:
                key = f"{mt}_{subset_name}"
                if key in all_best_params:
                    cnst = mc_catboost if mt == 'CatBoost' else mc_list
                    final_configs[mt] = {
                        'params': all_best_params[key],
                        'constraints': cnst,
                    }

            if final_configs:
                stack = StackingEnsemble(base_configs=final_configs, meta_alpha=1.0)
                X_np = X.values
                y_np = y.values
                stack.fit(X_np, y_np)
                stack_path = os.path.join(model_dir, f"Stacking_{subset_name}.pkl")
                stack.save(stack_path)
    else:
        print("\n[6/8] Skipping stacking ensemble (--skip-stacking)")

    # ── STEP 7: Gaussian Process ────────────────────────────────────────
    if not (args.skip_gp or args.quick):
        print("\n[7/8] Evaluating Gaussian Process regression...")

        for subset_name, subset_df in subsets.items():
            print(f"\n  GP for {subset_name}...")
            X = subset_df[feature_cols]
            y = subset_df['Compressive_Strength']

            gp_results = evaluate_gp_cv(X, y, n_folds=5, kernel_type='matern')

            print(f"\n  GP {subset_name} Results:")
            print(f"    RMSE: {gp_results['RMSE_mean']:.3f} ± {gp_results['RMSE_std']:.3f}")
            print(f"    R²:   {gp_results['R2_mean']:.4f} ± {gp_results['R2_std']:.4f}")
            print(f"    90% Coverage: {gp_results['Coverage90_mean']:.1f}%")
            print(f"    Mean Interval Width: {gp_results['MeanIntervalWidth']:.2f} MPa")

            all_results.append({
                'Subset': subset_name, 'Model': 'GaussianProcess',
                'RMSE_mean': gp_results['RMSE_mean'],
                'RMSE_std': gp_results['RMSE_std'],
                'MAE_mean': gp_results['MAE_mean'],
                'MAE_std': gp_results['MAE_std'],
                'R2_mean': gp_results['R2_mean'],
                'R2_std': gp_results['R2_std'],
            })

            # Save final GP model
            gp_model = create_gp_model('matern')
            gp_model.fit(X.values, y.values)
            gp_path = os.path.join(model_dir, f"GP_{subset_name}.pkl")
            save_gp_model(gp_model, gp_path)
    else:
        print("\n[7/8] Skipping GP (--skip-gp or --quick)")

    # ── STEP 8: Summary & comparison ────────────────────────────────────
    print("\n[8/8] Generating final summary...")

    results_df = pd.DataFrame(all_results).round(4)

    # Save results
    results_path = os.path.join(output_dir, 'stage_b_results_summary.csv')
    results_df.to_csv(results_path, index=False)
    print(f"\n  Results saved to {results_path}")

    # Save hyperparameters
    params_path = os.path.join(output_dir, 'best_hyperparameters_stage_b.json')
    with open(params_path, 'w') as f:
        json.dump(all_best_params, f, indent=2)
    print(f"  Hyperparameters saved to {params_path}")

    # Save feature list
    features_path = os.path.join(output_dir, 'feature_columns.json')
    with open(features_path, 'w') as f:
        json.dump(feature_cols, f, indent=2)
    print(f"  Feature columns saved to {features_path}")

    # Final summary
    total_time = time.time() - pipeline_start
    print(f"\n{'=' * 80}")
    print(f"  STAGE B RESULTS  (Total time: {total_time / 60:.1f} min)")
    print(f"{'=' * 80}")
    print(results_df.to_string(index=False))

    # Best model per subset
    print(f"\n{'─' * 60}")
    print("  Best model per subset (by R²):")
    print(f"{'─' * 60}")
    for subset in results_df['Subset'].unique():
        subset_results = results_df[results_df['Subset'] == subset]
        best_row = subset_results.loc[subset_results['R2_mean'].idxmax()]
        print(f"  {subset:<6}: {best_row['Model']:<25} "
              f"R²={best_row['R2_mean']:.4f}  "
              f"RMSE={best_row['RMSE_mean']:.3f} MPa")

    # Stage A comparison (if available)
    stage_a_path = os.path.join('outputs', 'stage_a_results_summary.csv')
    if os.path.exists(stage_a_path):
        print(f"\n{'─' * 60}")
        print("  Stage A vs Stage B Comparison:")
        print(f"{'─' * 60}")
        stage_a_df = pd.read_csv(stage_a_path)
        for subset in results_df['Subset'].unique():
            # Stage A best
            a_sub = stage_a_df[stage_a_df['Subset'] == subset]
            if len(a_sub) > 0:
                a_best = a_sub.loc[a_sub['R2_mean'].idxmax()]
                # Stage B best
                b_sub = results_df[results_df['Subset'] == subset]
                b_best = b_sub.loc[b_sub['R2_mean'].idxmax()]
                delta_r2 = b_best['R2_mean'] - a_best['R2_mean']
                delta_rmse = b_best['RMSE_mean'] - a_best['RMSE_mean']
                print(f"  {subset:<6}: A={a_best['R2_mean']:.4f} ({a_best['Model']}) "
                      f"→ B={b_best['R2_mean']:.4f} ({b_best['Model']})  "
                      f"ΔR²={delta_r2:+.4f}  ΔRMSE={delta_rmse:+.3f}")

    print(f"\n{'=' * 80}")
    print(f"  Pipeline complete! Models: '{model_dir}/'  Results: '{output_dir}/'")
    print(f"{'=' * 80}")

    return results_df


if __name__ == '__main__':
    main()
