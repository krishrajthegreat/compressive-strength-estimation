# Stage B Training & Inference Guide

## Prerequisites

```bash
cd compressive-strength-estimation
pip install -r requirements.txt
```

No new dependencies required — Stage B uses sklearn's GP (no PyTorch/BoTorch).

---

## Quick Validation (10 minutes)

Run a smoke test to verify all new code works:

```bash
python main_stage_b.py --quick
```

This runs:
- Full subset only (1,005 samples)
- 10 Optuna trials (vs 100 in full run)
- Skips GP (to save time)
- Tests: feature engineering (30 features), monotonic constraints, stacking

**Expected output:** Results in `outputs_stage_b/`, models in `models_stage_b/`

---

## Full Training Run (~6-10 hours)

### Option A: Everything (recommended for paper)

```bash
python main_stage_b.py --n-trials 100
```

This runs ALL subsets (EA1, EA7, EA14, Full) × ALL models + stacking + GP.

### Option B: Skip GP (saves ~1 hour)

```bash
python main_stage_b.py --n-trials 100 --skip-gp
```

### Option C: Individual models only (skip stacking, ~4 hours)

```bash
python main_stage_b.py --n-trials 100 --skip-stacking --skip-gp
```

### Option D: Custom trials

```bash
python main_stage_b.py --n-trials 50
```

---

## What Gets Saved

### Model Weights (`models_stage_b/`)

| File | Description |
|------|-------------|
| `XGBoost_Full.pkl` | XGBoost with monotonic constraints (Full) |
| `CatBoost_Full.pkl` | CatBoost with monotonic constraints (Full) |
| `LightGBM_Full.pkl` | LightGBM with monotonic constraints (Full) |
| `Stacking_Full.pkl` | Stacking ensemble: 3 GBDTs + Ridge (Full) |
| `GP_Full.pkl` | Gaussian Process model (Full) |
| `XGBoost_EA1.pkl` | XGBoost constrained (EA1) |
| `CatBoost_EA1.pkl` | CatBoost constrained (EA1) |
| `LightGBM_EA1.pkl` | LightGBM constrained (EA1) |
| `Stacking_EA1.pkl` | Stacking ensemble (EA1) |
| ... | Same pattern for EA7, EA14 |

### Results (`outputs_stage_b/`)

| File | Description |
|------|-------------|
| `stage_b_results_summary.csv` | All metrics: RMSE, MAE, R², per model × subset |
| `best_hyperparameters_stage_b.json` | Best Optuna params per model × subset |
| `feature_columns.json` | Ordered list of 30 feature column names |

---

## Inference (Loading Saved Models)

### Individual Model

```python
import pickle
import pandas as pd
from src.feature_engineering import engineer_features

# Load model
with open('models_stage_b/CatBoost_Full.pkl', 'rb') as f:
    model = pickle.load(f)

# Prepare input (raw mix design)
data = pd.DataFrame([{
    'Cement': 350, 'Blast_Furnace_Slag': 100, 'Fly_Ash': 50,
    'Water': 180, 'Superplasticizer': 8, 'Coarse_Aggregate': 1000,
    'Fine_Aggregate': 750, 'Age': 28,
    'Compressive_Strength': 0,  # dummy
}])

# Engineer features
data_feat = engineer_features(data)
feature_cols = [c for c in data_feat.columns if c != 'Compressive_Strength']
X = data_feat[feature_cols]

# Predict
prediction = model.predict(X)
print(f"Predicted strength: {prediction[0]:.2f} MPa")
```

### Stacking Ensemble

```python
from src.ensemble import StackingEnsemble

stack = StackingEnsemble.load('models_stage_b/Stacking_Full.pkl')
prediction = stack.predict(X.values)
print(f"Stacked prediction: {prediction[0]:.2f} MPa")
```

### GP with Uncertainty

```python
from src.gp_model import load_gp_model, gp_predict_with_uncertainty

gp = load_gp_model('models_stage_b/GP_Full.pkl')
mean, std = gp_predict_with_uncertainty(gp, X.values)
print(f"GP prediction: {mean[0]:.2f} ± {std[0]:.2f} MPa (68% CI)")
print(f"GP prediction: {mean[0]:.2f} ± {1.96*std[0]:.2f} MPa (95% CI)")
```

---

## Comparing Stage A vs Stage B

After running Stage B, the pipeline automatically compares results if
`outputs/stage_a_results_summary.csv` exists from a prior Stage A run.

To manually compare:

```python
import pandas as pd

a = pd.read_csv('outputs/stage_a_results_summary.csv')
b = pd.read_csv('outputs_stage_b/stage_b_results_summary.csv')

# Best per subset
for subset in ['EA1', 'EA7', 'EA14', 'Full']:
    a_sub = a[a['Subset'] == subset]
    b_sub = b[b['Subset'] == subset]
    if len(a_sub) > 0 and len(b_sub) > 0:
        a_best = a_sub.loc[a_sub['R2_mean'].idxmax()]
        b_best = b_sub.loc[b_sub['R2_mean'].idxmax()]
        print(f"{subset}: Stage A R²={a_best['R2_mean']:.4f} ({a_best['Model']}) "
              f"→ Stage B R²={b_best['R2_mean']:.4f} ({b_best['Model']})")
```

---

## Mix Design Optimization (After Training)

```python
from src.feature_engineering import engineer_features
from src.mix_optimizer import optimize_mix
from src.ensemble import StackingEnsemble

# Load trained model
stack = StackingEnsemble.load('models_stage_b/Stacking_Full.pkl')

# Find optimal mix for 40 MPa at 28 days
results = optimize_mix(
    target_strength=40.0,
    target_age=28,
    model=stack,
    feature_engineer_fn=engineer_features,
    wb_range=(0.30, 0.55),
    n_results=3,
)

for i, r in enumerate(results):
    print(f"\nMix {i+1}:")
    print(f"  Cement:       {r['Cement']} kg/m³")
    print(f"  Slag:         {r['Slag']} kg/m³")
    print(f"  Fly Ash:      {r['Fly_Ash']} kg/m³")
    print(f"  Water:        {r['Water']} kg/m³")
    print(f"  W/B ratio:    {r['W_B_ratio']}")
    print(f"  Predicted:    {r['Predicted_Strength']} MPa")
    print(f"  Error:        ±{r['Error_to_Target']} MPa")
```

---

## File Summary

### New Files (Stage B)

| File | Lines | Purpose |
|------|-------|---------|
| `src/ensemble.py` | ~280 | Stacking ensemble + OOF + nested CV |
| `src/gp_model.py` | ~170 | GP regression + uncertainty |
| `src/uncertainty.py` | ~160 | Conformal prediction |
| `src/mix_optimizer.py` | ~160 | BO-inspired mix optimizer |
| `main_stage_b.py` | ~290 | Stage B orchestrator |

### Modified Files

| File | Change |
|------|--------|
| `src/feature_engineering.py` | 22 → 30 features (8 new physics features) |
| `src/optimization.py` | Added monotonic constraints + constraint-aware HPO |
| `src/validation.py` | Constraints threaded through nested CV |

### Unchanged Files

| File | Why |
|------|-----|
| `main.py` | Stage A preserved for comparison |
| `src/data_loader.py` | No changes needed |
| `src/explainability.py` | SHAP works with new features automatically |
| `src/visualization.py` | Plots work with new results automatically |
| `app/` | Streamlit app untouched (update after comparison) |
