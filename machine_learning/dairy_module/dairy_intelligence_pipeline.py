"""
Dairy Intelligence System - Enhanced Pipeline
============================================

Professional agriculture intelligence system for dairy production forecasting.
Automatically detects and processes all dairy datasets in data/dairy/,
trains multiple ML models with cross-validation, and provides production-ready outputs.

Key Features:
1. Automatic dataset detection in data/dairy/
2. Column standardization (Year, Milk Production, State, Country)
3. Data aggregation and preprocessing
4. Multiple regression models with TimeSeriesSplit
5. Model comparison and selection
6. Versioned model saving
7. Comprehensive reporting and visualization

Author: Dairy Intelligence System
Version: 1.0.0
"""

import pandas as pd
import numpy as np
import os
import glob
import pickle
import warnings
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional

from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# ==============================
# CONFIGURATION
# ==============================

DATA_DIR = "data/dairy"
MODEL_SAVE_DIR = "models"
REPORT_DIR = "dairy_module/reports"
PLOT_SAVE_DIR = "dairy_module/visualizations"

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(PLOT_SAVE_DIR, exist_ok=True)

# ==============================
# 1️⃣ AUTOMATIC DATASET DETECTION AND PROCESSING
# ==============================

def detect_and_process_datasets(data_dir: str = DATA_DIR) -> pd.DataFrame:
    """
    Automatically detect all CSV files in data directory and process them.
    """
    print("=" * 70)
    print("DAIRY INTELLIGENCE SYSTEM - AUTOMATIC DATA PROCESSING")
    print("=" * 70)
    print(f"Data Directory: {data_dir}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Detect CSV files
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    # Filter out empty files
    csv_files = [f for f in csv_files if os.path.getsize(f) > 0]
    
    print(f"[DETECTION] Found {len(csv_files)} datasets:")
    for i, file_path in enumerate(csv_files, 1):
        filename = os.path.basename(file_path)
        size_kb = os.path.getsize(file_path) / 1024
        print(f"   {i}. {filename} ({size_kb:.1f} KB)")
    
    if not csv_files:
        raise FileNotFoundError("No CSV files found in data directory")
    
    # Process each dataset
    processed_datasets = []
    
    for file_path in csv_files:
        filename = os.path.basename(file_path)
        print(f"\n[PROCESSING] {filename}")
        
        try:
            # Load dataset
            df = pd.read_csv(file_path)
            print(f"   Raw shape: {df.shape}")
            print(f"   Columns: {list(df.columns)}")
            
            # Standardize column names
            df = standardize_columns(df, filename)
            
            # Validate required columns
            if 'Year' not in df.columns:
                print(f"   [SKIP] No 'Year' column found")
                continue
            
            if 'Milk_Production' not in df.columns:
                print(f"   [SKIP] No 'Milk_Production' column found")
                continue
            
            # Data type conversion and cleaning
            df = preprocess_dataset(df)
            
            if df is not None and not df.empty:
                processed_datasets.append(df[['Year', 'Milk_Production']])
                print(f"   [OK] Processed shape: {df.shape}")
            else:
                print(f"   [SKIP] Dataset is empty after processing")
                
        except Exception as e:
            print(f"   [ERROR] Failed to process {filename}: {str(e)}")
            continue
    
    if not processed_datasets:
        raise ValueError("No valid datasets processed successfully")
    
    # Aggregate datasets
    final_data = aggregate_datasets(processed_datasets)
    
    print("\n" + "=" * 70)
    print("[SUCCESS] Data Processing Complete!")
    print("=" * 70)
    print(f"Final Dataset: {final_data.shape[0]} rows × {final_data.shape[1]} columns")
    print(f"Year Range: {final_data['Year'].min()} - {final_data['Year'].max()}")
    print()
    
    return final_data

def standardize_columns(df: pd.DataFrame, filename: str) -> pd.DataFrame:
    """
    Standardize column names across different datasets.
    """
    print(f"   Standardizing columns...")
    
    col_mapping = {}
    
    # Standard column mappings
    for col in df.columns:
        col_lower = col.strip().lower()
        
        # Year column
        if col_lower in ['year', 'years']:
            col_mapping[col] = 'Year'
        
        # Milk Production column
        elif any(keyword in col_lower for keyword in ['milk', 'production', 'prod']):
            if 'milk' in col_lower and 'production' in col_lower:
                col_mapping[col] = 'Milk_Production'
            elif col_lower in ['production', 'production_pounds', 'value', 'prod']:
                col_mapping[col] = 'Milk_Production'
        
        # State column
        elif col_lower in ['state', 'states']:
            col_mapping[col] = 'State'
        
        # Country column
        elif col_lower in ['country', 'countries']:
            col_mapping[col] = 'Country'
        
        # Month column
        elif col_lower in ['month', 'months']:
            col_mapping[col] = 'Month'
    
    # Apply mapping
    df = df.rename(columns=col_mapping)
    
    if col_mapping:
        print(f"     Mapped columns: {col_mapping}")
    else:
        print(f"     No standard columns found. Available: {list(df.columns)}")
    
    return df

def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess a single dataset.
    """
    # Data type conversion
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    if 'Milk_Production' in df.columns:
        df['Milk_Production'] = pd.to_numeric(df['Milk_Production'], errors='coerce')
    
    # Remove invalid rows
    df = df.dropna(subset=['Year', 'Milk_Production'])
    
    if df.empty:
        return None
    
    df['Year'] = df['Year'].astype(int)
    
    # Handle Month column if present
    if 'Month' in df.columns and 'Year' in df.columns:
        try:
            df['Month'] = pd.to_datetime(df['Month'])
            df['Year'] = df['Month'].dt.year
        except:
            pass
    
    # Sort by Year
    df = df.sort_values('Year').reset_index(drop=True)
    
    # Remove duplicates (keep first occurrence)
    initial_rows = len(df)
    df = df.drop_duplicates(subset=['Year'], keep='first')
    if len(df) < initial_rows:
        print(f"   Removed {initial_rows - len(df)} duplicate rows")
    
    # Basic statistics
    print(f"   Year range: {df['Year'].min()} - {df['Year'].max()}")
    print(f"   Production range: {df['Milk_Production'].min():.0f} - {df['Milk_Production'].max():.0f}")
    
    return df

def aggregate_datasets(dataframes: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Aggregate multiple datasets by Year.
    """
    print(f"\n[AGGREGATION] Combining {len(dataframes)} datasets...")
    
    # Concatenate all datasets
    combined_df = pd.concat(dataframes, ignore_index=True)
    print(f"   Combined shape: {combined_df.shape}")
    
    # Group by Year and sum Milk Production (handles overlapping years)
    aggregated = combined_df.groupby('Year')['Milk_Production'].sum().reset_index()
    aggregated = aggregated.sort_values('Year').reset_index(drop=True)
    
    print(f"   Aggregated shape: {aggregated.shape}")
    print(f"   Final Year range: {aggregated['Year'].min()} - {aggregated['Year'].max()}")
    
    return aggregated

# ==============================
# 2. MODEL DEFINITION
# ==============================

def initialize_models() -> Dict[str, object]:
    """
    Initialize all regression models.
    """
    print("[INIT] Initializing machine learning models...")
    
    models = {
        # Linear Models
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0, random_state=42),
        "Lasso Regression": Lasso(alpha=0.1, random_state=42),
        
        # Polynomial Models
        "Polynomial (deg=2)": Pipeline([
            ("poly", PolynomialFeatures(degree=2)),
            ("scaler", StandardScaler()),
            ("lr", LinearRegression())
        ]),
        
        "Polynomial (deg=3)": Pipeline([
            ("poly", PolynomialFeatures(degree=3)),
            ("scaler", StandardScaler()),
            ("lr", LinearRegression())
        ]),
        
        # Tree-based Models
        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=12,
            min_samples_split=2,
            random_state=42,
            n_jobs=-1
        ),
        
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=300,
            max_depth=12,
            random_state=42,
            n_jobs=-1
        ),
        
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=4,
            random_state=42
        ),
        
        # Support Vector Regression
        "SVR": Pipeline([
            ("scaler", StandardScaler()),
            ("svr", SVR(kernel="rbf", C=100, epsilon=0.1))
        ])
    }
    
    print(f"[OK] {len(models)} models initialized")
    return models

# ==============================
# 3. MODEL TRAINING AND EVALUATION
# ==============================

def train_and_evaluate_models(df: pd.DataFrame, models: Dict[str, object]) -> Dict[str, Dict]:
    """
    Train all models using TimeSeriesSplit cross-validation.
    """
    print("=" * 70)
    print("MODEL TRAINING & EVALUATION")
    print("=" * 70)
    
    # Prepare features and target
    X = df[['Year']].values
    y = df['Milk_Production'].values
    
    print(f"Training on {len(X)} samples")
    print(f"Feature range: {X.min()} - {X.max()}")
    print(f"Target range: {y.min():.0f} - {y.max():.0f}")
    print(f"\nUsing TimeSeriesSplit (5 folds) for cross-validation")
    print("-" * 70)
    
    # TimeSeriesSplit for time-series data
    tscv = TimeSeriesSplit(n_splits=5)
    results = {}
    
    # Train and evaluate each model
    for name, model in models.items():
        print(f"\n>> {name}")
        
        try:
            # Cross-validation
            cv_scores = cross_val_score(model, X, y, cv=tscv, scoring="r2")
            cv_mae = -cross_val_score(model, X, y, cv=tscv, scoring="neg_mean_absolute_error")
            cv_rmse = np.sqrt(-cross_val_score(model, X, y, cv=tscv, scoring="neg_mean_squared_error"))
            
            mean_r2 = np.mean(cv_scores)
            std_r2 = np.std(cv_scores)
            mean_mae = np.mean(cv_mae)
            mean_rmse = np.mean(cv_rmse)
            
            # Full training for predictions
            model.fit(X, y)
            y_pred = model.predict(X)
            
            # Calculate full metrics
            full_r2 = r2_score(y, y_pred)
            full_mae = mean_absolute_error(y, y_pred)
            full_rmse = np.sqrt(mean_squared_error(y, y_pred))
            
            # Calculate MAPE (Mean Absolute Percentage Error)
            nonzero = y != 0
            mape = np.mean(np.abs((y[nonzero] - y_pred[nonzero]) / y[nonzero])) * 100
            
            # Store results
            results[name] = {
                "model": model,
                "cv_r2_mean": mean_r2,
                "cv_r2_std": std_r2,
                "cv_mae_mean": mean_mae,
                "cv_rmse_mean": mean_rmse,
                "full_r2": full_r2,
                "full_mae": full_mae,
                "full_rmse": full_rmse,
                "mape": mape,
                "predictions": y_pred,
                "cv_scores": cv_scores
            }
            
            print(f"   CV R² Score: {mean_r2:.4f} (±{std_r2:.4f})")
            print(f"   CV MAE: {mean_mae:.2f}")
            print(f"   CV RMSE: {mean_rmse:.2f}")
            print(f"   Full R²: {full_r2:.4f}")
            print(f"   MAE: {full_mae:.2f}")
            print(f"   RMSE: {full_rmse:.2f}")
            print(f"   MAPE: {mape:.2f}%")
            
        except Exception as e:
            print(f"   [ERROR] Failed to train {name}: {str(e)}")
            continue
    
    return results

def select_best_model(results: Dict[str, Dict]) -> Tuple[str, Dict, object]:
    """
    Select the best performing model based on CV R² score.
    """
    if not results:
        raise ValueError("No models trained successfully")
    
    sorted_results = sorted(results.items(), key=lambda x: x[1]["cv_r2_mean"], reverse=True)
    best_model_name = sorted_results[0][0]
    best_metrics = sorted_results[0][1]
    best_model = sorted_results[0][1]["model"]
    
    print("\n" + "=" * 70)
    print("MODEL COMPARISON RESULTS")
    print("=" * 70)
    print(f"\n{'Rank':<6} {'Model':<30} {'CV R²':<14} {'Full R²':<10} {'MAE':<10} {'RMSE':<10}")
    print("-" * 90)
    
    for rank, (name, metrics) in enumerate(sorted_results, 1):
        marker = "[BEST]" if rank == 1 else "      "
        print(f"{marker} {rank:<4} {name:<30} {metrics['cv_r2_mean']:.4f}±{metrics['cv_r2_std']:.3f}  "
              f"{metrics['full_r2']:<10.4f} {metrics['full_mae']:<10.2f} {metrics['full_rmse']:<10.2f}")
    
    return best_model_name, best_metrics, best_model

# ==============================
# 4. MODEL SAVING
# ==============================

def save_model_with_metadata(model: object, model_name: str, metrics: Dict, 
                           data_shape: Tuple, year_range: Tuple) -> str:
    """
    Save the best model with versioning and metadata.
    """
    # Generate version
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    version = f"v1_{timestamp}"
    
    # Save model
    model_filename = f"dairy_intelligence_{version}.pkl"
    model_path = os.path.join(MODEL_SAVE_DIR, model_filename)
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    # Save metadata (exclude the model object which is not JSON serializable)
    metadata = {
        "model_name": model_name,
        "version": version,
        "metrics": {
            "cv_r2_mean": float(metrics["cv_r2_mean"]),
            "cv_r2_std": float(metrics["cv_r2_std"]),
            "cv_mae_mean": float(metrics["cv_mae_mean"]),
            "cv_rmse_mean": float(metrics["cv_rmse_mean"]),
            "full_r2": float(metrics["full_r2"]),
            "full_mae": float(metrics["full_mae"]),
            "full_rmse": float(metrics["full_rmse"]),
            "mape": float(metrics["mape"])
        },
        "timestamp": datetime.now().isoformat(),
        "data_shape": data_shape,
        "year_range": year_range
    }
    
    metadata_path = os.path.join(MODEL_SAVE_DIR, f"dairy_metadata_{version}.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"[OK] Model saved: {model_path}")
    print(f"[OK] Metadata saved: {metadata_path}")
    
    return model_path

# ==============================
# 5. REPORT GENERATION
# ==============================

def generate_comprehensive_report(df: pd.DataFrame, results: Dict, best_model_name: str, 
                                best_metrics: Dict, model_path: str) -> str:
    """
    Generate comprehensive system report.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(REPORT_DIR, f"dairy_intelligence_report_{timestamp}.txt")
    
    report = []
    report.append("=" * 80)
    report.append("DAIRY INTELLIGENCE SYSTEM - COMPREHENSIVE REPORT")
    report.append("=" * 80)
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # Data Summary
    report.append("1. DATA PROCESSING SUMMARY")
    report.append("-" * 40)
    report.append(f"Final Dataset Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    report.append(f"Year Range: {df['Year'].min()} - {df['Year'].max()}")
    report.append(f"Production Range: {df['Milk_Production'].min():.0f} - {df['Milk_Production'].max():.0f}")
    report.append("")
    
    # Model Performance
    report.append("2. MODEL PERFORMANCE RESULTS")
    report.append("-" * 40)
    report.append(f"Best Model: {best_model_name}")
    report.append(f"Cross-Validation R² Score: {best_metrics['cv_r2_mean']:.6f} (±{best_metrics['cv_r2_std']:.6f})")
    report.append(f"Full Dataset R² Score: {best_metrics['full_r2']:.6f}")
    report.append(f"Mean Absolute Error: {best_metrics['full_mae']:.4f}")
    report.append(f"Root Mean Squared Error: {best_metrics['full_rmse']:.4f}")
    report.append(f"Mean Absolute Percentage Error: {best_metrics['mape']:.2f}%")
    report.append("")
    
    # Individual Model Comparison
    report.append("3. MODEL COMPARISON TABLE")
    report.append("-" * 40)
    report.append(f"{'Model':<35} {'CV R²':<14} {'Full R²':<10} {'MAE':<10} {'RMSE':<10}")
    report.append("-" * 90)
    sorted_results = sorted(results.items(), key=lambda x: x[1]["cv_r2_mean"], reverse=True)
    for name, metrics in sorted_results:
        report.append(f"{name:<35} {metrics['cv_r2_mean']:.4f}±{metrics['cv_r2_std']:.3f}  "
                     f"{metrics['full_r2']:<10.4f} {metrics['full_mae']:<10.2f} {metrics['full_rmse']:<10.2f}")
    report.append("")
    
    # Future Predictions
    report.append("4. FUTURE PROJECTIONS (2023-2030)")
    report.append("-" * 40)
    future_years = np.array([[year] for year in range(2023, 2031)])
    future_predictions = best_metrics['model'].predict(future_years)
    for year, pred in zip(future_years.flatten(), future_predictions):
        report.append(f"   {year}: {pred:.0f} units")
    report.append("")
    
    # Save report
    with open(report_path, 'w') as f:
        f.write('\n'.join(str(line) for line in report))
    
    print(f"[OK] Report saved: {report_path}")
    return report_path

# ==============================
# 6. VISUALIZATION
# ==============================

def generate_visualizations(df: pd.DataFrame, results: Dict, best_model_name: str, 
                          best_metrics: Dict) -> str:
    """
    Generate comprehensive visualizations.
    """
    print("\nGenerating visualizations...")
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (16, 12)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Dairy Intelligence System - {best_model_name}', 
                fontsize=16, fontweight='bold')
    
    # Plot 1: Actual vs Predicted
    ax1 = axes[0, 0]
    X = df[['Year']].values
    y = df['Milk_Production'].values
    ax1.scatter(X, y, color='blue', alpha=0.6, s=100, label='Actual', edgecolors='black')
    ax1.plot(X, best_metrics['predictions'], color='red', linewidth=2, label='Predicted')
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
    model_names = list(results.keys())
    cv_scores = [results[name]['cv_r2_mean'] for name in model_names]
    colors = ['green' if name == best_model_name else 'skyblue' for name in model_names]
    bars = ax3.barh(model_names, cv_scores, color=colors, edgecolor='black')
    ax3.set_xlabel('Cross-Validation R² Score', fontsize=12)
    ax3.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax3.set_xlim(0, 1)
    for i, (bar, score) in enumerate(zip(bars, cv_scores)):
        ax3.text(score + 0.01, i, f'{score:.4f}', va='center', fontsize=10)
    ax3.grid(True, alpha=0.3, axis='x')
    
    # Plot 4: Future Forecast
    ax4 = axes[1, 1]
    future_years = np.array([[year] for year in range(2023, 2031)])
    future_predictions = best_metrics['model'].predict(future_years)
    ax4.plot(X, y, 'o-', color='blue', linewidth=2, markersize=6, label='Historical Data')
    ax4.plot(future_years, future_predictions, 'o--', color='orange', linewidth=2, markersize=6, label='Forecast')
    ax4.set_xlabel('Year', fontsize=12)
    ax4.set_ylabel('Milk Production', fontsize=12)
    ax4.set_title('Production Forecast (2023-2030)', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Save plot
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = os.path.join(PLOT_SAVE_DIR, f"dairy_analysis_{timestamp}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Visualization saved: {plot_path}")
    
    return plot_path

# ==============================
# 7. MAIN PIPELINE EXECUTION
# ==============================

def run_dairy_intelligence_pipeline():
    """
    Run the complete Dairy Intelligence System pipeline.
    """
    print("=" * 80)
    print("DAIRY INTELLIGENCE SYSTEM - FULL PIPELINE EXECUTION")
    print("=" * 80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        # Step 1: Process data
        df = detect_and_process_datasets()
        
        # Step 2: Initialize models
        models = initialize_models()
        
        # Step 3: Train models
        results = train_and_evaluate_models(df, models)
        
        # Step 4: Select best model
        best_model_name, best_metrics, best_model = select_best_model(results)
        
        # Step 5: Save best model
        model_path = save_model_with_metadata(
            best_model, 
            best_model_name, 
            best_metrics,
            df.shape,
            (int(df['Year'].min()), int(df['Year'].max()))
        )
        
        # Step 6: Generate report
        report_path = generate_comprehensive_report(df, results, best_model_name, best_metrics, model_path)
        
        # Step 7: Generate visualizations
        plot_path = generate_visualizations(df, results, best_model_name, best_metrics)
        
        # Summary
        print("\n" + "=" * 80)
        print("PIPELINE EXECUTION COMPLETE!")
        print("=" * 80)
        print(f"Best Model: {best_model_name}")
        print(f"Cross-Validation R²: {best_metrics['cv_r2_mean']:.6f}")
        print(f"Model Saved: {model_path}")
        print(f"Report Generated: {report_path}")
        print(f"Visualization: {plot_path}")
        print("=" * 80)
        
        return {
            "model_path": model_path,
            "report_path": report_path,
            "plot_path": plot_path,
            "best_model": best_model_name,
            "cv_r2_score": best_metrics['cv_r2_mean']
        }
        
    except Exception as e:
        print(f"\n[ERROR] Pipeline failed: {str(e)}")
        raise

# Example usage
if __name__ == "__main__":
    # Run full pipeline
    results = run_dairy_intelligence_pipeline()
    
    # Access individual components
    print(f"\nFuture predictions using {results['best_model']}:")
    # Load the saved model
    with open(results['model_path'], 'rb') as f:
        model = pickle.load(f)
    
    future_years = np.array([[year] for year in range(2023, 2026)])
    predictions = model.predict(future_years)
    for year, pred in zip(future_years.flatten(), predictions):
        print(f"  {year}: {pred:.0f} units")