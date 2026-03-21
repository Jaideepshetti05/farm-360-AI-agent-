"""
=======================================================================
Agriculture Intelligence System — Dairy Module
=======================================================================
Comprehensive ML Training Pipeline for Milk Production Prediction

Auto-detects all CSV datasets in data/dairy/, standardises columns,
engineers features, trains 14+ regression models, evaluates with
TimeSeriesSplit cross-validation, saves versioned models, and produces
a full evaluation summary with visualisations.

Models trained:
  Classical   — Linear, Ridge, Lasso, ElasticNet
  Polynomial  — degree 2, 3, 4
  Tree-Based  — Decision Tree, Random Forest, Extra Trees,
                 Gradient Boosting, XGBoost (optional)
  SVR         — RBF kernel
  Neural      — MLPRegressor
  Ensemble    — VotingRegressor, StackingRegressor (built from best)

Author : Dairy Module – Agriculture Intelligence System
Version: 2.0
=======================================================================
"""

import os
import sys
import glob
import pickle
import warnings
import json
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    VotingRegressor,
    StackingRegressor,
)
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# Try optional imports
# ──────────────────────────────────────────────
try:
    from xgboost import XGBRegressor

    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

# ======================================================================
# CONFIGURATION
# ======================================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "dairy")
MODEL_DIR = os.path.join(BASE_DIR, "models")
EXPERIMENT_DIR = os.path.join(BASE_DIR, "dairy_module", "experiments")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(EXPERIMENT_DIR, exist_ok=True)

RANDOM_STATE = 42
CV_SPLITS = 5

# ======================================================================
# 1️  DATA LOADING — auto-detect every CSV in data/dairy/
# ======================================================================

def load_all_datasets(data_dir: str) -> pd.DataFrame:
    """
    Scan *data_dir* for CSV files, load each one, standardise column
    names to {Year, Milk_Production}, aggregate state/country-level rows
    by year, and return a single merged DataFrame sorted by Year.
    """
    csv_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not csv_files:
        sys.exit(f"[ERROR] No CSV files found in {data_dir}")

    frames = []
    dataset_info = []

    for path in csv_files:
        name = os.path.basename(path)
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            print(f"  [SKIP] {name}: cannot read ({exc})")
            continue

        if df.empty or len(df) < 2:
            print(f"  [SKIP] {name}: empty or too few rows ({len(df)})")
            continue

        # ── Standardise column names ──────────────────────────
        col_map = {}
        for c in df.columns:
            cl = c.strip().lower()
            if cl in ("year",):
                col_map[c] = "Year"
            elif "milk" in cl and "prod" in cl:
                col_map[c] = "Milk_Production"
            elif cl in ("production", "production_pounds", "value"):
                col_map[c] = "Milk_Production"
            elif cl in ("month",):
                col_map[c] = "Month"
            elif cl in ("state",):
                col_map[c] = "State"
            elif cl in ("country",):
                col_map[c] = "Country"

        df.rename(columns=col_map, inplace=True)

        # ── If there is a Month column but no Year, extract Year ──
        if "Month" in df.columns and "Year" not in df.columns:
            try:
                df["Month"] = pd.to_datetime(df["Month"])
                df["Year"] = df["Month"].dt.year
            except Exception:
                pass

        # ── Must have Year & Milk_Production ─────────────────
        if "Year" not in df.columns or "Milk_Production" not in df.columns:
            print(f"  [SKIP] {name}: missing Year/Milk_Production after mapping "
                  f"(cols: {list(df.columns)})")
            continue

        # ── Coerce types ─────────────────────────────────────
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
        df["Milk_Production"] = pd.to_numeric(df["Milk_Production"], errors="coerce")
        df.dropna(subset=["Year", "Milk_Production"], inplace=True)
        df["Year"] = df["Year"].astype(int)

        # ── Aggregate by Year (sum state / country / monthly rows) ──
        df_agg = df.groupby("Year", as_index=False)["Milk_Production"].sum()

        info = {
            "file": name,
            "raw_rows": len(df),
            "years": f"{int(df_agg['Year'].min())}-{int(df_agg['Year'].max())}",
            "agg_rows": len(df_agg),
        }
        dataset_info.append(info)
        frames.append(df_agg)
        print(f"  [OK]   {name}: {info['raw_rows']} rows → {info['agg_rows']} yearly "
              f"({info['years']})")

    if not frames:
        sys.exit("[ERROR] No usable datasets found!")

    # ── Merge all frames: average overlapping years ───────
    merged = pd.concat(frames, ignore_index=True)
    merged = merged.groupby("Year", as_index=False)["Milk_Production"].mean()
    merged.sort_values("Year", inplace=True)
    merged.reset_index(drop=True, inplace=True)

    return merged, dataset_info


# ======================================================================
# 2️  FEATURE ENGINEERING
# ======================================================================

def engineer_features(df: pd.DataFrame) -> tuple:
    """
    Create regression features from the Year column:
      - Year (raw)
      - Year_sq, Year_cu  (polynomial trends)
      - Decade            (categorical-ish decade bucket)
      - Years_Since_Start (normalised trend)
    Returns (X, y, feature_names).
    """
    df = df.copy()
    start = df["Year"].min()
    df["Years_Since_Start"] = df["Year"] - start
    df["Year_sq"] = df["Years_Since_Start"] ** 2
    df["Year_cu"] = df["Years_Since_Start"] ** 3
    df["Decade"] = (df["Year"] // 10) * 10

    feature_cols = ["Year", "Years_Since_Start", "Year_sq", "Year_cu", "Decade"]
    X = df[feature_cols].values
    y = df["Milk_Production"].values
    return X, y, feature_cols


# ======================================================================
# 3️  MODEL DEFINITIONS
# ======================================================================

def get_all_models() -> dict:
    """Return a dict  {name: estimator}  with every model to train."""
    models = {
        # ── Classical ────────────────────────────
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0, random_state=RANDOM_STATE),
        "Lasso Regression": Lasso(alpha=0.1, max_iter=10000, random_state=RANDOM_STATE),
        "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000,
                                 random_state=RANDOM_STATE),
        # ── Polynomial ───────────────────────────
        "Polynomial (deg=2)": Pipeline([
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("scaler", StandardScaler()),
            ("lr", LinearRegression()),
        ]),
        "Polynomial (deg=3)": Pipeline([
            ("poly", PolynomialFeatures(degree=3, include_bias=False)),
            ("scaler", StandardScaler()),
            ("lr", LinearRegression()),
        ]),
        "Polynomial (deg=4)": Pipeline([
            ("poly", PolynomialFeatures(degree=4, include_bias=False)),
            ("scaler", StandardScaler()),
            ("lr", LinearRegression()),
        ]),
        # ── Tree-Based ────────────────────────────
        "Decision Tree": DecisionTreeRegressor(max_depth=8, random_state=RANDOM_STATE),
        "Random Forest": RandomForestRegressor(
            n_estimators=300, max_depth=12, min_samples_split=2,
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=300, max_depth=12,
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=400, learning_rate=0.05, max_depth=4,
            subsample=0.8, random_state=RANDOM_STATE,
        ),
        # ── SVR ──────────────────────────────────
        "SVR (RBF)": Pipeline([
            ("scaler", StandardScaler()),
            ("svr", SVR(kernel="rbf", C=100, epsilon=0.1)),
        ]),
        # ── Neural ───────────────────────────────
        "MLP Regressor": Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPRegressor(
                hidden_layer_sizes=(128, 64),
                activation="relu",
                max_iter=2000,
                early_stopping=True,
                random_state=RANDOM_STATE,
            )),
        ]),
    }

    # ── XGBoost (optional) ───────────────────────
    if HAS_XGBOOST:
        models["XGBoost"] = XGBRegressor(
            n_estimators=400, learning_rate=0.05, max_depth=4, subsample=0.8,
            random_state=RANDOM_STATE, verbosity=0,
        )

    return models


# ======================================================================
# 4️  TRAINING & CROSS-VALIDATION
# ======================================================================

def evaluate_models(X, y, models: dict, cv_splits: int = CV_SPLITS) -> dict:
    """
    For each model:
      1. TimeSeriesSplit cross-validation  → CV R², CV MAE, CV RMSE
      2. Fit on full data                  → Full R², MAE, RMSE, MAPE
    Returns results dict keyed by model name.
    """
    tscv = TimeSeriesSplit(n_splits=cv_splits)
    results = {}

    for name, model in models.items():
        print(f"\n>> {name}")
        try:
            # Cross-validation
            cv_r2 = cross_val_score(model, X, y, cv=tscv, scoring="r2")
            cv_mae = -cross_val_score(model, X, y, cv=tscv,
                                      scoring="neg_mean_absolute_error")
            cv_rmse = np.sqrt(-cross_val_score(model, X, y, cv=tscv,
                                               scoring="neg_mean_squared_error"))

            # Full fit
            model.fit(X, y)
            y_pred = model.predict(X)
            full_r2 = r2_score(y, y_pred)
            full_mae = mean_absolute_error(y, y_pred)
            full_rmse = np.sqrt(mean_squared_error(y, y_pred))
            # MAPE (avoid division by zero)
            nonzero = y != 0
            mape = np.mean(np.abs((y[nonzero] - y_pred[nonzero]) / y[nonzero])) * 100

            results[name] = {
                "model": model,
                "cv_r2_mean": np.mean(cv_r2),
                "cv_r2_std": np.std(cv_r2),
                "cv_mae_mean": np.mean(cv_mae),
                "cv_rmse_mean": np.mean(cv_rmse),
                "full_r2": full_r2,
                "full_mae": full_mae,
                "full_rmse": full_rmse,
                "mape": mape,
                "predictions": y_pred,
            }
            print(f"   CV R²:   {np.mean(cv_r2):.4f} (±{np.std(cv_r2):.4f})")
            print(f"   Full R²: {full_r2:.4f}  |  MAE: {full_mae:.2f}  |  "
                  f"RMSE: {full_rmse:.2f}  |  MAPE: {mape:.2f}%")

        except Exception as exc:
            print(f"   [FAIL] {exc}")

    return results


# ======================================================================
# 5️  ENSEMBLE BUILDING — from top-3 models
# ======================================================================

def build_ensembles(results: dict, X, y) -> dict:
    """
    Build VotingRegressor and StackingRegressor from the top-3 CV R² models.
    Returns dict of ensemble results (same schema as evaluate_models).
    """
    sorted_models = sorted(results.items(), key=lambda x: x[1]["cv_r2_mean"],
                           reverse=True)
    top3 = sorted_models[:3]
    estimators = [(name.replace(" ", "_"), res["model"]) for name, res in top3]

    ensembles = {
        "VotingRegressor (Top-3)": VotingRegressor(estimators=estimators),
        "StackingRegressor (Top-3)": StackingRegressor(
            estimators=estimators,
            final_estimator=Ridge(alpha=1.0),
            cv=TimeSeriesSplit(n_splits=3),
        ),
    }

    tscv = TimeSeriesSplit(n_splits=CV_SPLITS)
    ens_results = {}
    for name, model in ensembles.items():
        print(f"\n>> {name}")
        try:
            cv_r2 = cross_val_score(model, X, y, cv=tscv, scoring="r2")
            cv_mae = -cross_val_score(model, X, y, cv=tscv,
                                      scoring="neg_mean_absolute_error")
            cv_rmse = np.sqrt(-cross_val_score(model, X, y, cv=tscv,
                                               scoring="neg_mean_squared_error"))
            model.fit(X, y)
            y_pred = model.predict(X)
            full_r2 = r2_score(y, y_pred)
            full_mae = mean_absolute_error(y, y_pred)
            full_rmse = np.sqrt(mean_squared_error(y, y_pred))
            nonzero = y != 0
            mape = np.mean(np.abs((y[nonzero] - y_pred[nonzero]) / y[nonzero])) * 100

            ens_results[name] = {
                "model": model,
                "cv_r2_mean": np.mean(cv_r2),
                "cv_r2_std": np.std(cv_r2),
                "cv_mae_mean": np.mean(cv_mae),
                "cv_rmse_mean": np.mean(cv_rmse),
                "full_r2": full_r2,
                "full_mae": full_mae,
                "full_rmse": full_rmse,
                "mape": mape,
                "predictions": y_pred,
            }
            print(f"   CV R²:   {np.mean(cv_r2):.4f} (±{np.std(cv_r2):.4f})")
            print(f"   Full R²: {full_r2:.4f}  |  MAE: {full_mae:.2f}  |  "
                  f"RMSE: {full_rmse:.2f}  |  MAPE: {mape:.2f}%")
        except Exception as exc:
            print(f"   [FAIL] {exc}")

    return ens_results


# ======================================================================
# 6️  HYPERPARAMETER TUNING — best model
# ======================================================================

PARAM_GRIDS = {
    "Random Forest": {
        "n_estimators": [200, 400, 600],
        "max_depth": [6, 10, 14, None],
        "min_samples_split": [2, 4, 6],
    },
    "Gradient Boosting": {
        "n_estimators": [200, 400, 600],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [3, 4, 6],
        "subsample": [0.7, 0.8, 1.0],
    },
    "Extra Trees": {
        "n_estimators": [200, 400, 600],
        "max_depth": [6, 10, 14, None],
        "min_samples_split": [2, 4],
    },
}

if HAS_XGBOOST:
    PARAM_GRIDS["XGBoost"] = {
        "n_estimators": [200, 400, 600],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [3, 4, 6],
        "subsample": [0.7, 0.8, 1.0],
    }


def tune_best_model(best_name: str, best_model, X, y) -> object:
    """If a param grid exists for the best model, run RandomizedSearchCV."""
    if best_name not in PARAM_GRIDS:
        print(f"   No tuning grid for '{best_name}'; skipping hyperparameter search.")
        return best_model

    print(f"   Running RandomizedSearchCV for '{best_name}'...")
    tscv = TimeSeriesSplit(n_splits=CV_SPLITS)
    search = RandomizedSearchCV(
        best_model, PARAM_GRIDS[best_name],
        n_iter=20, cv=tscv, scoring="r2",
        random_state=RANDOM_STATE, n_jobs=-1, verbose=0,
    )
    search.fit(X, y)
    print(f"   Best params: {search.best_params_}")
    print(f"   Tuned CV R²: {search.best_score_:.4f}")
    return search.best_estimator_


# ======================================================================
# 7️  MODEL SAVING — versioned, never overwrites
# ======================================================================

def next_version(model_dir: str, prefix: str = "dairy_regression") -> str:
    """Return next versioned filename, e.g. dairy_regression_v3.pkl"""
    existing = glob.glob(os.path.join(model_dir, f"{prefix}_v*.pkl"))
    if not existing:
        return os.path.join(model_dir, f"{prefix}_v1.pkl")
    versions = []
    for p in existing:
        base = os.path.splitext(os.path.basename(p))[0]
        try:
            v = int(base.split("_v")[-1])
            versions.append(v)
        except ValueError:
            pass
    nxt = max(versions) + 1 if versions else 1
    return os.path.join(model_dir, f"{prefix}_v{nxt}.pkl")


def save_model(model, model_dir: str) -> str:
    path = next_version(model_dir)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    return path


# ======================================================================
# 8️  VISUALISATION
# ======================================================================

def generate_visualisations(
    df: pd.DataFrame, y, results: dict, best_name: str, best_preds, experiment_dir: str
) -> str:
    """Generate a 2×3 figure with six analysis panels; return save path."""
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 3, figsize=(22, 13))
    fig.suptitle("Dairy Module — Model Analysis Dashboard", fontsize=18, fontweight="bold",
                 y=0.98)

    years = df["Year"].values

    # ---- 1. Actual vs Predicted (best model) ----
    ax = axes[0, 0]
    ax.scatter(years, y, c="#2563eb", s=60, alpha=0.7, edgecolors="k", label="Actual")
    ax.plot(years, best_preds, c="#ef4444", lw=2.5, label=f"Predicted ({best_name})")
    ax.set_xlabel("Year"); ax.set_ylabel("Milk Production")
    ax.set_title("Actual vs Predicted", fontweight="bold"); ax.legend(); ax.grid(alpha=0.3)

    # ---- 2. Residuals ----
    ax = axes[0, 1]
    residuals = y - best_preds
    ax.scatter(best_preds, residuals, c="#8b5cf6", s=60, alpha=0.7, edgecolors="k")
    ax.axhline(0, c="red", ls="--", lw=2)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Residual")
    ax.set_title("Residual Plot", fontweight="bold"); ax.grid(alpha=0.3)

    # ---- 3. Model Comparison (CV R²) ----
    ax = axes[0, 2]
    sorted_res = sorted(results.items(), key=lambda x: x[1]["cv_r2_mean"], reverse=True)
    names = [n for n, _ in sorted_res]
    scores = [m["cv_r2_mean"] for _, m in sorted_res]
    colors = ["#22c55e" if n == best_name else "#60a5fa" for n in names]
    bars = ax.barh(names, scores, color=colors, edgecolor="k", height=0.6)
    for bar, s in zip(bars, scores):
        ax.text(s + 0.005, bar.get_y() + bar.get_height() / 2, f"{s:.4f}",
                va="center", fontsize=8)
    ax.set_xlabel("CV R² Score"); ax.set_title("Model Comparison", fontweight="bold")
    ax.set_xlim(min(0, min(scores) - 0.05), 1.05); ax.grid(alpha=0.3, axis="x")

    # ---- 4. MAE Comparison ----
    ax = axes[1, 0]
    maes = [m["cv_mae_mean"] for _, m in sorted_res]
    ax.barh(names, maes, color=["#f97316" if n == best_name else "#fbbf24" for n in names],
            edgecolor="k", height=0.6)
    ax.set_xlabel("CV MAE"); ax.set_title("MAE Comparison", fontweight="bold")
    ax.grid(alpha=0.3, axis="x")

    # ---- 5. Forecast (next 8 years) ─────
    ax = axes[1, 1]
    best_model = results[best_name]["model"]
    last_year = int(years.max())
    future_years_raw = list(range(last_year + 1, last_year + 9))
    # Build feature array matching engineer_features logic
    start = int(df["Year"].min())
    future_df = pd.DataFrame({"Year": future_years_raw})
    future_df["Years_Since_Start"] = future_df["Year"] - start
    future_df["Year_sq"] = future_df["Years_Since_Start"] ** 2
    future_df["Year_cu"] = future_df["Years_Since_Start"] ** 3
    future_df["Decade"] = (future_df["Year"] // 10) * 10
    X_future = future_df[["Year", "Years_Since_Start", "Year_sq", "Year_cu", "Decade"]].values
    future_preds = best_model.predict(X_future)

    ax.plot(years, y, "o-", c="#2563eb", lw=2, ms=5, label="Historical")
    ax.plot(future_years_raw, future_preds, "o--", c="#f97316", lw=2, ms=7, label="Forecast")
    ax.set_xlabel("Year"); ax.set_ylabel("Milk Production")
    ax.set_title(f"Forecast ({last_year+1}–{last_year+8})", fontweight="bold")
    ax.legend(); ax.grid(alpha=0.3)

    # ---- 6. Residual Distribution ─────
    ax = axes[1, 2]
    ax.hist(residuals, bins=15, color="#a78bfa", edgecolor="k", alpha=0.85)
    ax.axvline(0, c="red", ls="--", lw=2)
    ax.set_xlabel("Residual"); ax.set_ylabel("Frequency")
    ax.set_title("Residual Distribution", fontweight="bold"); ax.grid(alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(experiment_dir, f"full_analysis_{ts}.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


# ======================================================================
# 9️  SUMMARY REPORT
# ======================================================================

def save_summary_report(
    dataset_info,
    feature_names,
    results: dict,
    best_name: str,
    best_metrics: dict,
    model_path: str,
    plot_path: str,
    experiment_dir: str,
) -> str:
    """Write a JSON + printed summary report."""
    sorted_res = sorted(results.items(), key=lambda x: x[1]["cv_r2_mean"], reverse=True)

    report = {
        "timestamp": datetime.now().isoformat(),
        "module": "Dairy Module — Agriculture Intelligence System",
        "datasets_loaded": dataset_info,
        "features_used": feature_names,
        "num_features": len(feature_names),
        "models_trained": len(results),
        "best_model": {
            "name": best_name,
            "cv_r2": round(best_metrics["cv_r2_mean"], 6),
            "cv_r2_std": round(best_metrics["cv_r2_std"], 6),
            "full_r2": round(best_metrics["full_r2"], 6),
            "mae": round(best_metrics["full_mae"], 4),
            "rmse": round(best_metrics["full_rmse"], 4),
            "mape_pct": round(best_metrics["mape"], 4),
            "saved_to": model_path,
        },
        "all_models": [
            {
                "rank": i + 1,
                "name": n,
                "cv_r2": round(m["cv_r2_mean"], 6),
                "full_r2": round(m["full_r2"], 6),
                "mae": round(m["full_mae"], 4),
                "rmse": round(m["full_rmse"], 4),
            }
            for i, (n, m) in enumerate(sorted_res)
        ],
        "visualisation": plot_path,
    }

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(experiment_dir, f"summary_report_{ts}.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    # ── Pretty-print to console ──────────────────────
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY REPORT")
    print("=" * 70)
    print(f"  Datasets loaded     : {len(dataset_info)}")
    print(f"  Features used       : {len(feature_names)}  {feature_names}")
    print(f"  Models trained      : {len(results)}")
    print(f"  CV Splits           : {CV_SPLITS}  (TimeSeriesSplit)")
    print()
    print(f"  {'Rank':<5} {'Model':<30} {'CV R²':<14} {'Full R²':<10} "
          f"{'MAE':<10} {'RMSE':<10}")
    print("-" * 80)
    for i, (n, m) in enumerate(sorted_res):
        tag = " ★" if n == best_name else ""
        print(f"  {i+1:<5} {n:<30} {m['cv_r2_mean']:.4f}±{m['cv_r2_std']:.3f}"
              f"   {m['full_r2']:<10.4f} {m['full_mae']:<10.2f} {m['full_rmse']:<10.2f}{tag}")

    print()
    print(f"  ★ BEST MODEL : {best_name}")
    print(f"    CV R²      : {best_metrics['cv_r2_mean']:.6f}")
    print(f"    Full R²    : {best_metrics['full_r2']:.6f}")
    print(f"    MAE        : {best_metrics['full_mae']:.4f}")
    print(f"    RMSE       : {best_metrics['full_rmse']:.4f}")
    print(f"    MAPE       : {best_metrics['mape']:.2f}%")
    print(f"    Saved to   : {model_path}")
    print(f"    Report     : {report_path}")
    print(f"    Plot       : {plot_path}")
    print("=" * 70)

    return report_path


# ======================================================================
# MAIN
# ======================================================================

def main():
    print("=" * 70)
    print("  AGRICULTURE INTELLIGENCE SYSTEM — DAIRY MODULE v2.0")
    print("=" * 70)

    # ── 1. Load data ─────────────────────────────────
    print("\n[STEP 1/7] Loading & merging all datasets...")
    df, dataset_info = load_all_datasets(DATA_DIR)
    print(f"\n  Merged dataset: {len(df)} yearly records "
          f"({int(df['Year'].min())}–{int(df['Year'].max())})")

    # ── 2. Feature engineering ───────────────────────
    print("\n[STEP 2/7] Engineering features...")
    X, y, feature_names = engineer_features(df)
    print(f"  Features: {feature_names}")
    print(f"  X shape: {X.shape}  |  y shape: {y.shape}")

    # ── 3. Build models ──────────────────────────────
    print("\n[STEP 3/7] Initialising models...")
    models = get_all_models()
    print(f"  {len(models)} models ready")

    # ── 4. Train & evaluate ──────────────────────────
    print(f"\n[STEP 4/7] Training & {CV_SPLITS}-fold TimeSeriesSplit CV...")
    print("-" * 70)
    results = evaluate_models(X, y, models)

    # ── 5. Build ensembles from top-3 ────────────────
    print(f"\n[STEP 5/7] Building ensemble models from top-3...")
    print("-" * 70)
    ens_results = build_ensembles(results, X, y)
    results.update(ens_results)  # merge into main results

    # ── Select best ──────────────────────────────────
    sorted_all = sorted(results.items(), key=lambda x: x[1]["cv_r2_mean"], reverse=True)
    best_name = sorted_all[0][0]
    best_metrics = sorted_all[0][1]
    best_model = best_metrics["model"]

    print(f"\n  ★ Current best: {best_name}  (CV R² = {best_metrics['cv_r2_mean']:.4f})")

    # ── 6. Hyperparameter tuning ─────────────────────
    print(f"\n[STEP 6/7] Hyperparameter tuning for best model...")
    tuned_model = tune_best_model(best_name, best_model, X, y)
    # Re-evaluate tuned model
    tuned_model.fit(X, y)
    tuned_preds = tuned_model.predict(X)
    tuned_r2 = r2_score(y, tuned_preds)
    if tuned_r2 > best_metrics["full_r2"]:
        print(f"  Tuned model improves Full R²: {best_metrics['full_r2']:.4f} → {tuned_r2:.4f}")
        best_metrics["model"] = tuned_model
        best_metrics["full_r2"] = tuned_r2
        best_metrics["full_mae"] = mean_absolute_error(y, tuned_preds)
        best_metrics["full_rmse"] = np.sqrt(mean_squared_error(y, tuned_preds))
        best_metrics["predictions"] = tuned_preds
        results[best_name] = best_metrics

    # ── 7. Save & report ─────────────────────────────
    print(f"\n[STEP 7/7] Saving model & generating outputs...")
    model_path = save_model(best_metrics["model"], MODEL_DIR)
    print(f"  Model saved: {model_path}")

    plot_path = generate_visualisations(
        df, y, results, best_name, best_metrics["predictions"], EXPERIMENT_DIR
    )
    print(f"  Visualisation saved: {plot_path}")

    report_path = save_summary_report(
        dataset_info, feature_names, results, best_name, best_metrics,
        model_path, plot_path, EXPERIMENT_DIR,
    )

    print("\n[SUCCESS] Dairy Module training complete!")


if __name__ == "__main__":
    main()
