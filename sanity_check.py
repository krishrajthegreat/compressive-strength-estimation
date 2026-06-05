#!/usr/bin/env python3
"""
Stage B Sanity Check — validates all code works WITHOUT training.

Checks:
  1. All imports resolve
  2. Feature engineering produces 30 features
  3. Monotonic constraint builder works
  4. A pre-trained model (if available) can make predictions
  5. Stacking ensemble save/load round-trip
  6. GP model forward pass (2 samples)
  7. Conformal predictor calibration
  8. Mix optimizer runs

Run this BEFORE the full training run to catch code errors.

Usage:
    python sanity_check.py
"""

import sys
import os
import numpy as np
import pandas as pd
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PASS = "[PASS]"
FAIL = "[FAIL]"
SKIP = "[SKIP]"


def check(name, fn):
    try:
        result = fn()
        msg = f" -- {result}" if result else ""
        print(f"  {PASS} {name}{msg}")
        return True
    except Exception as e:
        print(f"  {FAIL} {name}")
        print(f"         Error: {e}")
        traceback.print_exc()
        return False


def main():
    print("=" * 60)
    print("  Stage B Sanity Check")
    print("=" * 60)

    passed = 0
    total = 0

    # ── 1. Imports ─────────────────────────────────────────────
    print("\n[1] Imports")

    def _import_fe():
        from src.feature_engineering import engineer_features, get_feature_columns
        return "ok"
    total += 1; passed += check("feature_engineering", _import_fe)

    def _import_opt():
        from src.optimization import (
            build_monotonic_constraints, build_catboost_constraints, MONOTONIC_MAP
        )
        return f"{len(MONOTONIC_MAP)} constrained features"
    total += 1; passed += check("optimization (constraints)", _import_opt)

    def _import_ens():
        from src.ensemble import StackingEnsemble, generate_oof_predictions
        return "ok"
    total += 1; passed += check("ensemble", _import_ens)

    def _import_gp():
        from src.gp_model import create_gp_model, gp_predict_with_uncertainty
        return "ok"
    total += 1; passed += check("gp_model", _import_gp)

    def _import_unc():
        from src.uncertainty import SplitConformalPredictor
        return "ok"
    total += 1; passed += check("uncertainty", _import_unc)

    def _import_opt2():
        from src.mix_optimizer import optimize_mix
        return "ok"
    total += 1; passed += check("mix_optimizer", _import_opt2)

    # ── 2. Data loading ────────────────────────────────────────
    print("\n[2] Data Loading & Feature Engineering")

    data_path = "data/Concrete_Data - Sheet1.csv"
    df = None

    def _load():
        nonlocal df
        from src.data_loader import load_and_preprocess_data
        df = load_and_preprocess_data(data_path)
        return f"{len(df)} rows"
    total += 1; passed += check(f"load_and_preprocess ('{data_path}')", _load)

    df_feat = None
    feature_cols = None

    def _features():
        nonlocal df_feat, feature_cols
        from src.feature_engineering import engineer_features, get_feature_columns
        df_feat = engineer_features(df, verbose=True)
        feature_cols = get_feature_columns(df_feat)
        assert len(feature_cols) == 30, f"Expected 30 features, got {len(feature_cols)}"
        return f"{len(feature_cols)} features"
    total += 1; passed += check("engineer_features (30 features)", _features)

    # ── 3. Monotonic constraints ───────────────────────────────
    print("\n[3] Monotonic Constraints")

    mc_list = None
    mc_cat = None

    def _mc_list():
        nonlocal mc_list
        from src.optimization import build_monotonic_constraints
        mc_list = build_monotonic_constraints(feature_cols)
        n_constrained = sum(1 for c in mc_list if c != 0)
        assert len(mc_list) == 30
        return f"{n_constrained} features constrained"
    total += 1; passed += check("build_monotonic_constraints", _mc_list)

    def _mc_cat():
        nonlocal mc_cat
        from src.optimization import build_catboost_constraints
        mc_cat = build_catboost_constraints(feature_cols)
        return f"{len(mc_cat)} CatBoost constraints"
    total += 1; passed += check("build_catboost_constraints", _mc_cat)

    # ── 4. GP forward pass (tiny) ──────────────────────────────
    print("\n[4] Gaussian Process (2-sample smoke test)")

    def _gp_smoke():
        from src.gp_model import create_gp_model, gp_predict_with_uncertainty
        gp = create_gp_model('matern')
        # Use 20 random samples to keep it fast
        rng = np.random.RandomState(42)
        X_tiny = df_feat[feature_cols].values[:20]
        y_tiny = df_feat['Compressive_Strength'].values[:20]
        gp.fit(X_tiny, y_tiny)
        mean, std = gp_predict_with_uncertainty(gp, X_tiny[:2])
        assert len(mean) == 2 and len(std) == 2
        return f"mean={mean[0]:.2f} std={std[0]:.2f}"
    total += 1; passed += check("GP fit + predict (20 samples)", _gp_smoke)

    # ── 5. Conformal predictor ─────────────────────────────────
    print("\n[5] Conformal Prediction")

    def _conformal():
        from src.uncertainty import SplitConformalPredictor
        import numpy as np
        cp = SplitConformalPredictor(alpha=0.10)
        y_cal = np.array([30, 35, 40, 45, 50], dtype=float)
        y_pred = np.array([28, 36, 38, 47, 51], dtype=float)
        cp.calibrate(y_cal, y_pred)
        lower, upper = cp.predict_interval(np.array([35.0]))
        assert lower[0] < 35.0 < upper[0]
        return f"interval width={cp.get_interval_width()*2:.2f} MPa"
    total += 1; passed += check("SplitConformalPredictor", _conformal)

    # ── 6. Stacking ensemble (tiny round-trip) ─────────────────
    print("\n[6] Stacking Ensemble (tiny round-trip)")

    def _stacking_smoke():
        from src.ensemble import StackingEnsemble
        import xgboost as xgb
        from catboost import CatBoostRegressor
        import lightgbm as lgb
        import tempfile

        X_tiny = df_feat[feature_cols].values[:50]
        y_tiny = df_feat['Compressive_Strength'].values[:50]

        base_configs = {
            'XGBoost': {
                'params': {'n_estimators': 10, 'max_depth': 3, 'learning_rate': 0.1},
                'constraints': mc_list,
            },
            'CatBoost': {
                'params': {'iterations': 10, 'depth': 3, 'learning_rate': 0.1},
                'constraints': mc_cat,
            },
            'LightGBM': {
                'params': {'n_estimators': 10, 'num_leaves': 15, 'learning_rate': 0.1},
                'constraints': mc_list,
            },
        }

        stack = StackingEnsemble(base_configs=base_configs, n_oof_splits=3)
        stack.fit(X_tiny, y_tiny)
        preds = stack.predict(X_tiny[:5])
        assert len(preds) == 5

        # Save/load round trip
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            tmp_path = f.name
        stack.save(tmp_path)
        loaded = StackingEnsemble.load(tmp_path)
        preds2 = loaded.predict(X_tiny[:5])
        os.unlink(tmp_path)
        assert np.allclose(preds, preds2)
        return f"stack pred={preds[0]:.2f}, save/load OK"
    total += 1; passed += check("StackingEnsemble fit+predict+save/load", _stacking_smoke)

    # ── 7. Mix optimizer ───────────────────────────────────────
    print("\n[7] Mix Optimizer")

    def _mix_opt():
        from src.mix_optimizer import optimize_mix
        from src.feature_engineering import engineer_features
        import xgboost as xgb

        X_tiny = df_feat[feature_cols].values[:100]
        y_tiny = df_feat['Compressive_Strength'].values[:100]

        # Use a quick XGBoost as surrogate
        surrogate = xgb.XGBRegressor(n_estimators=20, max_depth=3, random_state=42)
        surrogate.fit(X_tiny, y_tiny)

        # Use a silent wrapper for the optimizer
        def silent_engineer(df):
            return engineer_features(df, verbose=False)

        results = optimize_mix(
            target_strength=35.0,
            target_age=28,
            model=surrogate,
            feature_engineer_fn=silent_engineer,
            n_results=1,
            maxiter=2,
        )
        assert len(results) == 1
        return f"suggested mix: {results[0]['Predicted_Strength']:.1f} MPa"
    total += 1; passed += check("mix_optimizer (1 result)", _mix_opt)

    # ── Summary ────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  Result: {passed}/{total} checks passed")
    if passed == total:
        print("  ALL CHECKS PASSED -- ready for full training run")
    else:
        print(f"  {total - passed} check(s) FAILED -- fix before training")
    print("=" * 60)

    return passed == total


if __name__ == '__main__':
    ok = main()
    sys.exit(0 if ok else 1)
