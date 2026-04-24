"""
Agricultural AI Machine Learning System
=======================================

Professional system for automated training of all CSV datasets in agricultural domains.
Supports dairy production forecasting, animal classification, and health prediction.

Features:
- Automatic dataset detection across multiple directories
- Intelligent column standardization
- Comprehensive data preprocessing
- Multi-model training for regression and classification
- Professional evaluation and model selection
- Automated reporting and visualization

Author: Agricultural AI System
Version: 1.0.0
"""

import pandas as pd
import numpy as np
import os
import glob
import pickle
import json
import warnings
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path

# ML Libraries
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_regression, f_classif
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error, 
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report
)

# Regression Models
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import (
    RandomForestRegressor, ExtraTreesRegressor, 
    GradientBoostingRegressor, AdaBoostRegressor
)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

# Classification Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier,
    GradientBoostingClassifier, AdaBoostClassifier
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

# Try to import XGBoost
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("XGBoost not available. Install with: pip install xgboost")

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

class AgriculturalAISystem:
    """Main class for the Agricultural AI Machine Learning System."""
    
    def __init__(self, base_dir: str = ".", model_dir: str = "models"):
        """
        Initialize the system.
        
        Args:
            base_dir (str): Base project directory
            model_dir (str): Directory to save trained models
        """
        self.base_dir = Path(base_dir)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        # Create output directories
        (self.base_dir / "reports").mkdir(exist_ok=True)
        (self.base_dir / "visualizations").mkdir(exist_ok=True)
        
        # Configuration
        self.data_dirs = [
            "data/dairy",
            "data/animal_classification", 
            "data/health"
        ]
        
        # Column mapping standards
        self.column_standards = {
            'year': ['year', 'Year', 'YEAR', 'years', 'Years'],
            'date': ['date', 'Date', 'DATE'],
            'milk_production': [
                'milk production', 'Milk Production', 'MILK_PRODUCTION',
                'production', 'Production', 'PRODUCTION',
                'milk_prod', 'Milk_Prod', 'Milk_Yield_L', 'yield'
            ],
            'animal_type': [
                'animal', 'Animal', 'ANIMAL', 'animal_type', 'Animal_Type',
                'species', 'Species', 'breed', 'Breed', 'class_type'
            ],
            'health_status': [
                'health', 'Health', 'HEALTH', 'health_status', 'Health_Status',
                'disease', 'Disease', 'condition', 'Condition'
            ],
            'target': ['target', 'Target', 'TARGET', 'label', 'Label'],
            'state': ['state', 'State', 'STATE', 'states', 'States'],
            'country': ['country', 'Country', 'COUNTRY', 'countries', 'Countries']
        }
        
        # Task detection keywords
        self.regression_keywords = ['production', 'yield', 'milk', 'weight', 'temperature']
        self.classification_keywords = ['type', 'class', 'species', 'health', 'disease', 'status']
        
        # Results storage
        self.processed_datasets = {}
        self.trained_models = {}
        self.evaluation_results = {}
        
    def detect_datasets(self) -> Dict[str, List[str]]:
        """
        Automatically detect all CSV files in configured data directories.
        
        Returns:
            Dictionary mapping directory names to list of CSV file paths
        """
        print("=" * 80)
        print("AGRICULTURAL AI SYSTEM - DATASET DETECTION")
        print("=" * 80)
        
        datasets = {}
        
        for data_dir in self.data_dirs:
            dir_path = self.base_dir / data_dir
            if not dir_path.exists():
                print(f"[SKIP] Directory not found: {data_dir}")
                continue
                
            # Find all CSV files (non-empty)
            csv_files = [
                str(f) for f in dir_path.glob("*.csv") 
                if f.is_file() and f.stat().st_size > 0
            ]
            
            if csv_files:
                datasets[data_dir] = csv_files
                print(f"[DETECT] {data_dir}: {len(csv_files)} datasets found")
                for file in csv_files:
                    size_kb = os.path.getsize(file) / 1024
                    print(f"   - {os.path.basename(file)} ({size_kb:.1f} KB)")
            else:
                print(f"[INFO] No CSV files found in {data_dir}")
        
        if not datasets:
            raise FileNotFoundError("No datasets found in any configured directories")
            
        return datasets
    
    def standardize_columns(self, df: pd.DataFrame, filename: str) -> pd.DataFrame:
        """
        Standardize column names across different datasets.
        
        Args:
            df (pd.DataFrame): Input dataframe
            filename (str): Source filename for logging
            
        Returns:
            DataFrame with standardized column names
        """
        print(f"   Standardizing columns for {os.path.basename(filename)}...")
        
        col_mapping = {}
        
        # Map columns to standard names
        for col in df.columns:
            col_clean = str(col).strip()
            col_lower = col_clean.lower()
            
            # Check each standard category
            for standard_name, variants in self.column_standards.items():
                if col_lower in [v.lower() for v in variants] or \
                   any(keyword in col_lower for keyword in variants):
                    col_mapping[col] = standard_name
                    break
        
        # Apply mapping
        df_standardized = df.rename(columns=col_mapping)
        
        if col_mapping:
            print(f"     Mapped: {col_mapping}")
        else:
            print(f"     No standard mappings found. Columns: {list(df.columns)}")
            
        return df_standardized
    
    def detect_task_type(self, df: pd.DataFrame, filename: str) -> str:
        """
        Automatically detect if dataset is for regression or classification.
        
        Args:
            df (pd.DataFrame): Processed dataframe
            filename (str): Source filename
            
        Returns:
            Task type: 'regression' or 'classification'
        """
        # Check for target column
        if 'target' in df.columns:
            target_values = df['target'].dropna()
            unique_values = target_values.nunique()
            
            # Classification if discrete values (<= 20 unique values)
            if unique_values <= 20:
                print(f"   Task detected: CLASSIFICATION (target has {unique_values} unique values)")
                return 'classification'
            else:
                print(f"   Task detected: REGRESSION (target has {unique_values} unique values)")
                return 'regression'
        
        # Check filename for keywords
        filename_lower = filename.lower()
        
        # Regression keywords
        if any(keyword in filename_lower for keyword in self.regression_keywords):
            print(f"   Task detected: REGRESSION (filename keywords)")
            return 'regression'
            
        # Classification keywords
        if any(keyword in filename_lower for keyword in self.classification_keywords):
            print(f"   Task detected: CLASSIFICATION (filename keywords)")
            return 'classification'
        
        # Default to regression for numeric targets
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            print(f"   Task detected: REGRESSION (numeric columns present)")
            return 'regression'
        else:
            print(f"   Task detected: CLASSIFICATION (categorical data)")
            return 'classification'
    
    def preprocess_data(self, df: pd.DataFrame, task_type: str) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """
        Comprehensive data preprocessing pipeline.
        
        Args:
            df (pd.DataFrame): Input dataframe
            task_type (str): 'regression' or 'classification'
            
        Returns:
            Tuple of (features_X, target_y, feature_names)
        """
        print("   Preprocessing data...")
        
        df_processed = df.copy()
        
        # 1. Handle missing values
        print("     - Handling missing values...")
        numeric_columns = df_processed.select_dtypes(include=[np.number]).columns
        categorical_columns = df_processed.select_dtypes(exclude=[np.number]).columns
        
        # Impute numeric columns with median
        if len(numeric_columns) > 0:
            numeric_imputer = SimpleImputer(strategy='median')
            df_processed[numeric_columns] = numeric_imputer.fit_transform(df_processed[numeric_columns])
        
        # Impute categorical columns with mode
        if len(categorical_columns) > 0:
            categorical_imputer = SimpleImputer(strategy='most_frequent')
            df_processed[categorical_columns] = categorical_imputer.fit_transform(df_processed[categorical_columns])
        
        # 2. Handle duplicates
        initial_rows = len(df_processed)
        df_processed = df_processed.drop_duplicates()
        if len(df_processed) < initial_rows:
            print(f"     - Removed {initial_rows - len(df_processed)} duplicate rows")
        
        # 3. Identify target variable
        if 'target' in df_processed.columns:
            target_col = 'target'
        elif task_type == 'regression' and 'milk_production' in df_processed.columns:
            target_col = 'milk_production'
        elif task_type == 'classification' and 'class_type' in df_processed.columns:
            target_col = 'class_type'
        elif task_type == 'classification' and 'animal_type' in df_processed.columns:
            target_col = 'animal_type'
        elif task_type == 'classification' and 'health_status' in df_processed.columns:
            target_col = 'health_status'
        else:
            # Use last column as target if no standard target found
            target_col = df_processed.columns[-1]
            print(f"     - Using '{target_col}' as target variable")
        
        # 4. Separate features and target
        y = df_processed[target_col]
        X = df_processed.drop(columns=[target_col])
        
        # 5. Encode categorical variables
        categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()
        if categorical_features:
            print(f"     - Encoding {len(categorical_features)} categorical features")
            # Use Label Encoding for simplicity (can be extended to One-Hot)
            for col in categorical_features:
                try:
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))
                except Exception as e:
                    print(f"       Warning: Could not encode {col}: {str(e)}")
                    # Drop problematic columns
                    X = X.drop(columns=[col])
        
        # 6. Feature scaling for numeric features
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_features:
            print(f"     - Scaling {len(numeric_features)} numeric features")
            scaler = StandardScaler()
            try:
                X[numeric_features] = scaler.fit_transform(X[numeric_features])
            except Exception as e:
                print(f"       Warning: Scaling failed: {str(e)}")
        
        # 7. Remove outliers for regression tasks (using IQR method) - only for smaller datasets
        if task_type == 'regression' and len(numeric_features) > 0 and len(X) < 50000:
            print("     - Removing outliers (IQR method)")
            try:
                initial_rows = len(X)
                for col in numeric_features[:5]:  # Only check first 5 features to avoid memory issues
                    Q1 = X[col].quantile(0.25)
                    Q3 = X[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    mask = (X[col] >= lower_bound) & (X[col] <= upper_bound)
                    X = X[mask]
                    y = y[mask.reset_index(drop=True)]
                if len(X) < initial_rows:
                    print(f"     - Removed {initial_rows - len(X)} outlier rows")
            except Exception as e:
                print(f"       Warning: Outlier removal failed: {str(e)}")
        
        print(f"     - Final shape: {X.shape[0]} samples, {X.shape[1]} features")
        
        return X, y, X.columns.tolist()
    
    def get_regression_models(self) -> Dict[str, object]:
        """Get all regression models to train."""
        models = {
            "Linear Regression": LinearRegression(),
            "Ridge Regression": Ridge(alpha=1.0, random_state=42),
            "Lasso Regression": Lasso(alpha=0.1, random_state=42),
            "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42),
            "Polynomial Regression (deg=2)": Pipeline([
                ("poly", PolynomialFeatures(degree=2)),
                ("scaler", StandardScaler()),
                ("regressor", LinearRegression())
            ]),
            "Random Forest Regressor": RandomForestRegressor(
                n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
            ),
            "Extra Trees Regressor": ExtraTreesRegressor(
                n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
            ),
            "Gradient Boosting Regressor": GradientBoostingRegressor(
                n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42
            ),
            "AdaBoost Regressor": AdaBoostRegressor(
                n_estimators=100, learning_rate=0.1, random_state=42
            ),
            "Support Vector Regressor": SVR(kernel='rbf', C=1.0, epsilon=0.1),
            "K-Nearest Neighbors": KNeighborsRegressor(n_neighbors=5),
            "Decision Tree": DecisionTreeRegressor(max_depth=10, random_state=42)
        }
        
        # Add XGBoost if available
        if HAS_XGBOOST:
            models["XGBoost Regressor"] = xgb.XGBRegressor(
                n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42
            )
        
        return models
    
    def get_classification_models(self) -> Dict[str, object]:
        """Get all classification models to train."""
        models = {
            "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
            "Random Forest Classifier": RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
            ),
            "Extra Trees Classifier": ExtraTreesClassifier(
                n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
            ),
            "Gradient Boosting Classifier": GradientBoostingClassifier(
                n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42
            ),
            "AdaBoost Classifier": AdaBoostClassifier(
                n_estimators=100, learning_rate=0.1, random_state=42
            ),
            "Support Vector Machine": SVC(kernel='rbf', C=1.0, random_state=42, probability=True),
            "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
            "Naive Bayes": GaussianNB(),
            "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42)
        }
        
        # Add XGBoost if available
        if HAS_XGBOOST:
            models["XGBoost Classifier"] = xgb.XGBClassifier(
                n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42
            )
        
        return models
    
    def train_models(self, X: pd.DataFrame, y: pd.Series, task_type: str, 
                    dataset_name: str) -> Dict[str, Dict]:
        """
        Train all models for the given task.
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target variable
            task_type (str): 'regression' or 'classification'
            dataset_name (str): Name for reporting
            
        Returns:
            Dictionary of model evaluation results
        """
        print(f"\n>> Training {task_type} models for {dataset_name}")
        print(f"   Data shape: {X.shape[0]} samples × {X.shape[1]} features")
        
        # Select appropriate models
        if task_type == 'regression':
            models = self.get_regression_models()
            scoring = 'r2'
            cv_splitter = TimeSeriesSplit(n_splits=5) if 'year' in X.columns or 'date' in X.columns else 5
        else:
            models = self.get_classification_models()
            scoring = 'accuracy'
            cv_splitter = 5
        
        results = {}
        
        for name, model in models.items():
            try:
                print(f"   - {name}")
                
                # Cross-validation
                cv_scores = cross_val_score(model, X, y, cv=cv_splitter, scoring=scoring)
                
                # Train on full data
                model.fit(X, y)
                y_pred = model.predict(X)
                
                # Calculate metrics
                if task_type == 'regression':
                    metrics = {
                        'cv_score_mean': np.mean(cv_scores),
                        'cv_score_std': np.std(cv_scores),
                        'r2_score': r2_score(y, y_pred),
                        'mae': mean_absolute_error(y, y_pred),
                        'rmse': np.sqrt(mean_squared_error(y, y_pred)),
                        'mape': np.mean(np.abs((y - y_pred) / y)) * 100 if np.all(y != 0) else np.inf
                    }
                else:
                    metrics = {
                        'cv_score_mean': np.mean(cv_scores),
                        'cv_score_std': np.std(cv_scores),
                        'accuracy': accuracy_score(y, y_pred),
                        'precision': precision_score(y, y_pred, average='weighted', zero_division=0),
                        'recall': recall_score(y, y_pred, average='weighted', zero_division=0),
                        'f1_score': f1_score(y, y_pred, average='weighted', zero_division=0)
                    }
                    
                    # Add ROC-AUC for binary classification
                    if len(np.unique(y)) == 2:
                        try:
                            y_pred_proba = model.predict_proba(X)[:, 1]
                            metrics['roc_auc'] = roc_auc_score(y, y_pred_proba)
                        except:
                            metrics['roc_auc'] = None
                
                results[name] = {
                    'model': model,
                    'metrics': metrics,
                    'predictions': y_pred
                }
                
                # Print key metrics
                if task_type == 'regression':
                    print(f"     CV R²: {metrics['cv_score_mean']:.4f} ± {metrics['cv_score_std']:.4f}")
                    print(f"     MAE: {metrics['mae']:.2f}, RMSE: {metrics['rmse']:.2f}")
                else:
                    print(f"     CV Accuracy: {metrics['cv_score_mean']:.4f} ± {metrics['cv_score_std']:.4f}")
                    print(f"     F1: {metrics['f1_score']:.4f}")
                    
            except Exception as e:
                print(f"     [ERROR] {name} failed: {str(e)}")
                continue
        
        return results
    
    def select_best_model(self, results: Dict[str, Dict], task_type: str) -> Tuple[str, Dict]:
        """
        Select the best model based on cross-validation performance.
        
        Args:
            results (Dict): Model evaluation results
            task_type (str): 'regression' or 'classification'
            
        Returns:
            Tuple of (best_model_name, best_model_results)
        """
        if not results:
            raise ValueError("No models trained successfully")
        
        # Sort by cross-validation score
        if task_type == 'regression':
            sorted_results = sorted(results.items(), 
                                  key=lambda x: x[1]['metrics']['cv_score_mean'], 
                                  reverse=True)
        else:
            sorted_results = sorted(results.items(), 
                                  key=lambda x: x[1]['metrics']['cv_score_mean'], 
                                  reverse=True)
        
        best_name, best_results = sorted_results[0]
        
        print(f"\n[BEST MODEL] {best_name}")
        print(f"   CV Score: {best_results['metrics']['cv_score_mean']:.4f} ± {best_results['metrics']['cv_score_std']:.4f}")
        
        return best_name, best_results
    
    def save_model_and_metadata(self, model: object, model_name: str, metrics: Dict,
                              dataset_name: str, task_type: str, 
                              feature_names: List[str]) -> str:
        """
        Save model with versioned naming and metadata.
        
        Args:
            model (object): Trained model
            model_name (str): Name of the model
            metrics (Dict): Model performance metrics
            dataset_name (str): Source dataset name
            task_type (str): Task type
            feature_names (List[str]): Feature names used
            
        Returns:
            Path to saved model file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version = f"v1_{timestamp}"
        
        # Save model
        model_filename = f"agri_ai_{task_type}_{version}.pkl"
        model_path = self.model_dir / model_filename
        
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Save metadata
        metadata = {
            "model_name": model_name,
            "task_type": task_type,
            "dataset_name": dataset_name,
            "version": version,
            "metrics": {k: float(v) if isinstance(v, (int, float, np.integer, np.floating)) else v 
                       for k, v in metrics.items()},
            "feature_names": feature_names,
            "timestamp": datetime.now().isoformat(),
            "data_shape": {"samples": len(self.processed_datasets[dataset_name]['X']),
                          "features": len(feature_names)}
        }
        
        metadata_path = self.model_dir / f"metadata_{task_type}_{version}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"[SAVED] Model: {model_path}")
        print(f"[SAVED] Metadata: {metadata_path}")
        
        return str(model_path)
    
    def generate_report(self, dataset_name: str, task_type: str, results: Dict,
                       best_model_name: str, best_metrics: Dict, 
                       feature_names: List[str]) -> str:
        """
        Generate comprehensive training report.
        
        Args:
            dataset_name (str): Name of dataset
            task_type (str): Task type
            results (Dict): All model results
            best_model_name (str): Best model name
            best_metrics (Dict): Best model metrics
            feature_names (List[str]): Feature names
            
        Returns:
            Path to report file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.base_dir / "reports" / f"agri_ai_report_{timestamp}.txt"
        
        # Sort results by CV score
        sorted_results = sorted(results.items(), 
                              key=lambda x: x[1]['metrics']['cv_score_mean'], 
                              reverse=True)
        
        # Generate report content
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("AGRICULTURAL AI SYSTEM - TRAINING REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Dataset: {dataset_name}")
        report_lines.append(f"Task Type: {task_type.upper()}")
        report_lines.append(f"Features: {len(feature_names)}")
        report_lines.append(f"Feature Names: {', '.join(feature_names[:10])}{'...' if len(feature_names) > 10 else ''}")
        report_lines.append("")
        
        # Best Model Results
        report_lines.append("BEST MODEL PERFORMANCE")
        report_lines.append("-" * 40)
        report_lines.append(f"Model: {best_model_name}")
        if task_type == 'regression':
            report_lines.append(f"CV R² Score: {best_metrics['cv_score_mean']:.6f} ± {best_metrics['cv_score_std']:.6f}")
            report_lines.append(f"R² Score: {best_metrics['r2_score']:.6f}")
            report_lines.append(f"MAE: {best_metrics['mae']:.4f}")
            report_lines.append(f"RMSE: {best_metrics['rmse']:.4f}")
            report_lines.append(f"MAPE: {best_metrics['mape']:.2f}%")
        else:
            report_lines.append(f"CV Accuracy: {best_metrics['cv_score_mean']:.6f} ± {best_metrics['cv_score_std']:.6f}")
            report_lines.append(f"Accuracy: {best_metrics['accuracy']:.6f}")
            report_lines.append(f"Precision: {best_metrics['precision']:.6f}")
            report_lines.append(f"Recall: {best_metrics['recall']:.6f}")
            report_lines.append(f"F1-Score: {best_metrics['f1_score']:.6f}")
            if 'roc_auc' in best_metrics and best_metrics['roc_auc'] is not None:
                report_lines.append(f"ROC-AUC: {best_metrics['roc_auc']:.6f}")
        report_lines.append("")
        
        # Model Comparison
        report_lines.append("MODEL COMPARISON")
        report_lines.append("-" * 40)
        if task_type == 'regression':
            report_lines.append(f"{'Model':<30} {'CV R²':<12} {'R²':<10} {'MAE':<10}")
            report_lines.append("-" * 70)
            for name, res in sorted_results[:10]:  # Top 10 models
                metrics = res['metrics']
                report_lines.append(f"{name:<30} {metrics['cv_score_mean']:.4f}±{metrics['cv_score_std']:.3f}  "
                                  f"{metrics['r2_score']:<10.4f} {metrics['mae']:<10.2f}")
        else:
            report_lines.append(f"{'Model':<30} {'CV Acc':<12} {'Accuracy':<10} {'F1':<10}")
            report_lines.append("-" * 70)
            for name, res in sorted_results[:10]:  # Top 10 models
                metrics = res['metrics']
                report_lines.append(f"{name:<30} {metrics['cv_score_mean']:.4f}±{metrics['cv_score_std']:.3f}  "
                                  f"{metrics['accuracy']:<10.4f} {metrics['f1_score']:<10.4f}")
        
        # Save report
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        print(f"[REPORT] Saved to: {report_path}")
        return str(report_path)
    
    def run_full_pipeline(self):
        """
        Run the complete Agricultural AI pipeline for all datasets.
        """
        print("=" * 80)
        print("AGRICULTURAL AI MACHINE LEARNING SYSTEM")
        print("=" * 80)
        print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        try:
            # 1. Detect datasets
            datasets = self.detect_datasets()
            
            # 2. Process each dataset
            for data_dir, file_paths in datasets.items():
                print(f"\n{'='*60}")
                print(f"PROCESSING DATASETS IN: {data_dir}")
                print(f"{'='*60}")
                
                for file_path in file_paths:
                    try:
                        dataset_name = os.path.basename(file_path)
                        print(f"\n[DATASET] {dataset_name}")
                        
                        # Load data
                        df = pd.read_csv(file_path)
                        print(f"   Raw shape: {df.shape}")
                        
                        # Standardize columns
                        df_standardized = self.standardize_columns(df, file_path)
                        
                        # Detect task type
                        task_type = self.detect_task_type(df_standardized, dataset_name)
                        
                        # Preprocess data
                        X, y, feature_names = self.preprocess_data(df_standardized, task_type)
                        
                        # Store processed data
                        self.processed_datasets[dataset_name] = {
                            'X': X, 'y': y, 'task_type': task_type, 
                            'feature_names': feature_names
                        }
                        
                        # Train models
                        results = self.train_models(X, y, task_type, dataset_name)
                        
                        if not results:
                            print(f"   [SKIP] No models trained successfully")
                            continue
                        
                        # Select best model
                        best_name, best_results = self.select_best_model(results, task_type)
                        
                        # Save best model
                        model_path = self.save_model_and_metadata(
                            best_results['model'], best_name, best_results['metrics'],
                            dataset_name, task_type, feature_names
                        )
                        
                        # Generate report
                        report_path = self.generate_report(
                            dataset_name, task_type, results, 
                            best_name, best_results['metrics'], feature_names
                        )
                        
                        # Store results
                        self.trained_models[dataset_name] = {
                            'best_model': best_results['model'],
                            'model_path': model_path,
                            'report_path': report_path,
                            'metrics': best_results['metrics'],
                            'task_type': task_type
                        }
                        
                        print(f"   [COMPLETE] {dataset_name}")
                        
                    except Exception as e:
                        print(f"   [ERROR] Failed to process {file_path}: {str(e)}")
                        continue
            
            # Summary
            print(f"\n{'='*80}")
            print("PIPELINE EXECUTION COMPLETE")
            print(f"{'='*80}")
            print(f"Datasets processed: {len(self.processed_datasets)}")
            print(f"Models trained: {len(self.trained_models)}")
            print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            return self.trained_models
            
        except Exception as e:
            print(f"\n[ERROR] Pipeline failed: {str(e)}")
            raise

# Example usage
if __name__ == "__main__":
    # Initialize system
    system = AgriculturalAISystem()
    
    # Run full pipeline
    results = system.run_full_pipeline()
    
    # Print summary
    print(f"\nTrained models summary:")
    for dataset, info in results.items():
        print(f"  {dataset}: {info['task_type']} - {info['metrics']['cv_score_mean']:.4f}")
