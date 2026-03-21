"""
Comprehensive ML Training Pipeline for Dairy Production Data
==============================================================

This script trains multiple machine learning models on dairy production datasets
and selects the best performing model based on cross-validation scores.

Models trained:
- Linear Regression (baseline)
- Polynomial Regression (degree 2 and 3)
- Random Forest Regressor
- Gradient Boosting Regressor

Evaluation: TimeSeriesSplit cross-validation with R², MAE, RMSE metrics
"""

import pandas as pd
import numpy as np
import os
import pickle
import warnings
from datetime import datetime

from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# ==============================
# CONFIGURATION
# ==============================

DATA_PATH_YEARLY = "data/dairy/Mik_Pro.csv"
DATA_PATH_MONTHLY = "data/dairy/monthly-milk-production-pounds.csv"
MODEL_SAVE_DIR = "models"
PLOT_SAVE_DIR = "dairy_module/experiments"

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(PLOT_SAVE_DIR, exist_ok=True)

# ==============================
# 1️⃣ LOAD AND PREPARE DATA
# ==============================

print("=" * 60)
print("DAIRY PRODUCTION ML TRAINING PIPELINE")
print("=" * 60)
print(f"\nLoading datasets...")

# Load yearly data
df_yearly = pd.read_csv(DATA_PATH_YEARLY)
print(f"[OK] Yearly dataset: {len(df_yearly)} rows ({df_yearly['Year'].min()}-{df_yearly['Year'].max()})")

# Load monthly data
df_monthly = pd.read_csv(DATA_PATH_MONTHLY)
df_monthly['Month'] = pd.to_datetime(df_monthly['Month'])
df_monthly['Year'] = df_monthly['Month'].dt.year
df_monthly['MonthNum'] = df_monthly['Month'].dt.month
print(f"[OK] Monthly dataset: {len(df_monthly)} rows ({df_monthly['Year'].min()}-{df_monthly['Year'].max()})")

# Use yearly data for main training
X = df_yearly[['Year']].values
y = df_yearly['Milk Production'].values

print(f"\nTraining on yearly data: {len(X)} samples")
print(f"   Target range: {y.min():.0f} - {y.max():.0f}")

# ==============================
# 2. DEFINE MODELS
# ==============================

print(f"\nInitializing models...")

models = {
    "Linear Regression": LinearRegression(),
    
    "Polynomial Regression (deg=2)": Pipeline([
        ("poly", PolynomialFeatures(degree=2)),
        ("scaler", StandardScaler()),
        ("lr", LinearRegression())
    ]),
    
    "Polynomial Regression (deg=3)": Pipeline([
        ("poly", PolynomialFeatures(degree=3)),
        ("scaler", StandardScaler()),
        ("lr", LinearRegression())
    ]),
    
    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_split=2,
        random_state=42,
        n_jobs=-1
    ),
    
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )
}

print(f"[OK] {len(models)} models initialized")

# ==============================
# 3. CROSS-VALIDATION
# ==============================

print(f"\nRunning 5-fold TimeSeriesSplit cross-validation...")
print("-" * 60)

tscv = TimeSeriesSplit(n_splits=5)
results = {}

for name, model in models.items():
    print(f"\n>> {name}")
    
    # Cross-validation scores
    cv_scores = cross_val_score(model, X, y, cv=tscv, scoring="r2")
    mean_r2 = np.mean(cv_scores)
    std_r2 = np.std(cv_scores)
    
    # Train on full data for other metrics
    model.fit(X, y)
    y_pred = model.predict(X)
    
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2_full = r2_score(y, y_pred)
    
    results[name] = {
        "model": model,
        "cv_r2_mean": mean_r2,
        "cv_r2_std": std_r2,
        "r2": r2_full,
        "mae": mae,
        "rmse": rmse,
        "predictions": y_pred
    }
    
    print(f"   CV R² Score: {mean_r2:.4f} (±{std_r2:.4f})")
    print(f"   Full R² Score: {r2_full:.4f}")
    print(f"   MAE: {mae:.2f}")
    print(f"   RMSE: {rmse:.2f}")

# ==============================
# 4. SELECT BEST MODEL
# ==============================

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

# Sort by CV R² score
sorted_results = sorted(results.items(), key=lambda x: x[1]["cv_r2_mean"], reverse=True)

print(f"\n{'Rank':<6} {'Model':<30} {'CV R²':<12} {'R²':<10} {'MAE':<10} {'RMSE':<10}")
print("-" * 80)

for rank, (name, metrics) in enumerate(sorted_results, 1):
    print(f"{rank:<6} {name:<30} {metrics['cv_r2_mean']:.4f}±{metrics['cv_r2_std']:.3f}  "
          f"{metrics['r2']:.4f}    {metrics['mae']:<10.2f} {metrics['rmse']:<10.2f}")

best_model_name = sorted_results[0][0]
best_model = sorted_results[0][1]["model"]
best_metrics = sorted_results[0][1]

print(f"\n[BEST] Best Model: {best_model_name}")
print(f"   CV R² Score: {best_metrics['cv_r2_mean']:.4f}")
print(f"   Full R² Score: {best_metrics['r2']:.4f}")

# ==============================
# 5. SAVE BEST MODEL
# ==============================

print(f"\nSaving best model...")

model_path = os.path.join(MODEL_SAVE_DIR, "dairy_regression_model.pkl")
scaler_path = os.path.join(MODEL_SAVE_DIR, "dairy_scaler.pkl")

with open(model_path, 'wb') as f:
    pickle.dump(best_model, f)
print(f"[OK] Model saved: {model_path}")

# Save scaler if it's a pipeline with scaler
if hasattr(best_model, 'named_steps') and 'scaler' in best_model.named_steps:
    with open(scaler_path, 'wb') as f:
        pickle.dump(best_model.named_steps['scaler'], f)
    print(f"[OK] Scaler saved: {scaler_path}")

# ==============================
# 6. GENERATE VISUALIZATIONS
# ==============================

print(f"\nGenerating visualizations...")

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle(f'Dairy Production Model Analysis - {best_model_name}', fontsize=16, fontweight='bold')

# Plot 1: Actual vs Predicted
ax1 = axes[0, 0]
ax1.scatter(df_yearly['Year'], y, color='blue', alpha=0.6, s=100, label='Actual', edgecolors='black')
ax1.plot(df_yearly['Year'], best_metrics['predictions'], color='red', linewidth=2, label='Predicted')
ax1.set_xlabel('Year', fontsize=12)
ax1.set_ylabel('Milk Production', fontsize=12)
ax1.set_title('Actual vs Predicted Values', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Residuals
ax2 = axes[0, 1]
residuals = y - best_metrics['predictions']
ax2.scatter(best_metrics['predictions'], residuals, color='purple', alpha=0.6, s=100, edgecolors='black')
ax2.axhline(y=0, color='red', linestyle='--', linewidth=2)
ax2.set_xlabel('Predicted Values', fontsize=12)
ax2.set_ylabel('Residuals', fontsize=12)
ax2.set_title('Residual Plot', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)

# Plot 3: Model Comparison
ax3 = axes[1, 0]
model_names = [name for name, _ in sorted_results]
cv_scores = [metrics['cv_r2_mean'] for _, metrics in sorted_results]
colors = ['green' if i == 0 else 'skyblue' for i in range(len(model_names))]
bars = ax3.barh(model_names, cv_scores, color=colors, edgecolor='black')
ax3.set_xlabel('CV R² Score', fontsize=12)
ax3.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
ax3.set_xlim(0, 1)
for i, (bar, score) in enumerate(zip(bars, cv_scores)):
    ax3.text(score + 0.01, i, f'{score:.4f}', va='center', fontsize=10)
ax3.grid(True, alpha=0.3, axis='x')

# Plot 4: Forecast
ax4 = axes[1, 1]
future_years = np.array([[year] for year in range(2023, 2031)])
future_predictions = best_model.predict(future_years)
ax4.plot(df_yearly['Year'], y, 'o-', color='blue', linewidth=2, markersize=6, label='Historical Data')
ax4.plot(future_years, future_predictions, 'o--', color='orange', linewidth=2, markersize=6, label='Forecast')
ax4.set_xlabel('Year', fontsize=12)
ax4.set_ylabel('Milk Production', fontsize=12)
ax4.set_title('Production Forecast (2023-2030)', fontsize=14, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plot_path = os.path.join(PLOT_SAVE_DIR, f"model_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"[OK] Visualization saved: {plot_path}")

# ==============================
# 7. FUTURE PREDICTIONS
# ==============================

print(f"\nFuture Predictions (2023-2030):")
print("-" * 40)
for year, pred in zip(future_years.flatten(), future_predictions):
    print(f"   {year}: {pred:.0f}")

print("\n" + "=" * 60)
print("[SUCCESS] TRAINING COMPLETE!")
print("=" * 60)
