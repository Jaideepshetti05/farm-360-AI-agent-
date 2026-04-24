"""
Comprehensive Agricultural Dataset Training Pipeline
Trains models in ALL possible ways for agricultural production prediction
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')
import os
from datetime import datetime
import json

class AgricultureMLPipeline:
    """Comprehensive ML pipeline for agriculture dataset"""
    
    def __init__(self, data_path="data/agriculture/Production_Crops_Livestock_E_All_Data.csv"):
        self.data_path = data_path
        self.df = None
        self.processed_data = {}
        self.models = {}
        self.results = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create directories
        self.model_dir = f"models/agriculture_{self.timestamp}"
        self.report_dir = f"reports/agriculture_{self.timestamp}"
        self.vis_dir = f"visualizations/agriculture_{self.timestamp}"
        
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.report_dir, exist_ok=True)
        os.makedirs(self.vis_dir, exist_ok=True)
    
    def load_and_explore_data(self):
        """Load and explore the agriculture dataset"""
        print("Loading agriculture dataset...")
        self.df = pd.read_csv(self.data_path)
        
        print(f"Dataset shape: {self.df.shape}")
        print(f"Columns: {list(self.df.columns)}")
        print(f"Data types:\n{self.df.dtypes}")
        print(f"Missing values:\n{self.df.isnull().sum()}")
        
        # Display basic info
        print("\nFirst few rows:")
        print(self.df.head())
        
        # Identify year columns
        year_columns = [col for col in self.df.columns if col.startswith('Y')]
        print(f"\nYear columns found: {len(year_columns)}")
        print(f"Years range: {year_columns[0][1:]} to {year_columns[-1][1:]}")
        
        # Element types
        print(f"\nUnique Elements: {self.df['Element'].unique()}")
        print(f"Unique Items: {len(self.df['Item'].unique())}")
        print(f"Unique Areas: {len(self.df['Area'].unique())}")
        
        return self.df
    
    def preprocess_data(self):
        """Preprocess the agriculture dataset for different modeling approaches"""
        print("\nPreprocessing data for multiple approaches...")
        
        # Get year columns
        year_columns = [col for col in self.df.columns if col.startswith('Y')]
        
        # Approach 1: Time series - Focus on production over time for each country/crop
        production_df = self.df[self.df['Element'] == 'Production'].copy()
        if production_df.empty:
            # If 'Production' isn't found, look for similar terms
            elements = self.df['Element'].unique()
            print(f"Available elements: {elements}")
            # Use first available element that seems related to production
            production_df = self.df.copy()
        
        # Clean the data - convert year columns to numeric
        for col in year_columns:
            production_df[col] = pd.to_numeric(production_df[col], errors='coerce')
        
        # Remove rows where all year values are NaN
        production_df = production_df.dropna(subset=year_columns, how='all')
        
        # Approach 2: Cross-sectional - Predict production based on features
        features_df = production_df[['Area', 'Item', 'Element']].copy()
        
        # Create aggregate features
        features_df['mean_production'] = production_df[year_columns].mean(axis=1, skipna=True)
        features_df['std_production'] = production_df[year_columns].std(axis=1, skipna=True)
        features_df['min_production'] = production_df[year_columns].min(axis=1, skipna=True)
        features_df['max_production'] = production_df[year_columns].max(axis=1, skipna=True)
        
        # Encode categorical variables
        le_area = LabelEncoder()
        le_item = LabelEncoder()
        
        features_df['Area_encoded'] = le_area.fit_transform(features_df['Area'].fillna('Unknown'))
        features_df['Item_encoded'] = le_item.fit_transform(features_df['Item'].fillna('Unknown'))
        
        # Store encoders for later use
        self.area_encoder = le_area
        self.item_encoder = le_item
        
        # Store processed data
        self.processed_data['production_ts'] = production_df
        self.processed_data['features_df'] = features_df
        self.processed_data['year_columns'] = year_columns
        
        print(f"Production time series shape: {production_df.shape}")
        print(f"Features dataframe shape: {features_df.shape}")
        
        return features_df
    
    def train_regression_models(self):
        """Train regression models to predict agricultural production"""
        print("\n=== Training Regression Models ===")
        
        features_df = self.processed_data['features_df']
        
        # Prepare features and target
        feature_cols = ['Area_encoded', 'Item_encoded', 'std_production', 'min_production', 'max_production']
        X = features_df[feature_cols].dropna()
        y = features_df.loc[X.index, 'mean_production'].dropna()
        
        # Ensure X and y have same index
        X = X.loc[y.index]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Define models
        regression_models = {
            'Linear Regression': LinearRegression(),
            'Random Forest Regressor': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting Regressor': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'SVR': SVR(kernel='rbf'),
            'XGBoost': xgb.XGBRegressor(random_state=42),
            'LightGBM': lgb.LGBMRegressor(random_state=42)
        }
        
        reg_results = {}
        
        for name, model in regression_models.items():
            print(f"Training {name}...")
            
            # Fit model
            if name in ['SVR']:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            
            # Cross-validation
            if name in ['SVR']:
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
            else:
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
            
            reg_results[name] = {
                'mse': mse,
                'rmse': rmse,
                'r2_score': r2,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'predictions': y_pred.tolist(),
                'actual': y_test.tolist(),
                'model': model
            }
            
            print(f"  RMSE: {rmse:.4f}, R²: {r2:.4f}, CV R²: {cv_scores.mean():.4f}±{cv_scores.std():.4f}")
        
        # Store results
        self.models['regression'] = reg_results
        self.results['regression'] = reg_results
        
        # Save scaler
        import joblib
        joblib.dump(scaler, os.path.join(self.model_dir, 'regression_scaler.pkl'))
        
        return reg_results
    
    def train_time_series_forecasting(self):
        """Train time series forecasting models"""
        print("\n=== Training Time Series Forecasting Models ===")
        
        production_df = self.processed_data['production_ts']
        year_columns = self.processed_data['year_columns']
        
        # For simplicity, let's forecast the average production for top countries/crops
        # Aggregate by area and item to get average production over time
        ts_data = production_df.groupby(['Area', 'Item'])[year_columns].mean()
        
        # Select top 10 combinations for demonstration
        top_combinations = ts_data.mean(axis=1).nlargest(10).index
        
        ts_results = {}
        
        for area, item in top_combinations:
            print(f"Training time series model for {area} - {item}")
            
            # Get the time series
            series = ts_data.loc[(area, item)].dropna()
            
            if len(series) > 5:  # Need at least some data points
                # Prepare data for supervised learning
                X, y = [], []
                for i in range(len(series) - 3):  # Use 3 previous values to predict next
                    X.append([series.iloc[i], series.iloc[i+1], series.iloc[i+2]])
                    y.append(series.iloc[i+3])
                
                if len(X) > 0:
                    X = np.array(X)
                    y = np.array(y)
                    
                    # Split data
                    split_idx = int(0.8 * len(X))
                    X_train, X_test = X[:split_idx], X[split_idx:]
                    y_train, y_test = y[:split_idx], y[split_idx:]
                    
                    # Train models
                    models = {
                        'TS_Linear': LinearRegression(),
                        'TS_RF': RandomForestRegressor(n_estimators=50, random_state=42)
                    }
                    
                    model_results = {}
                    for name, model in models.items():
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        
                        mse = mean_squared_error(y_test, y_pred)
                        rmse = np.sqrt(mse)
                        r2 = r2_score(y_test, y_pred)
                        
                        model_results[name] = {
                            'mse': mse,
                            'rmse': rmse,
                            'r2_score': r2,
                            'predictions': y_pred.tolist(),
                            'actual': y_test.tolist()
                        }
                    
                    ts_results[f"{area}_{item}"] = model_results
        
        self.models['timeseries'] = ts_results
        self.results['timeseries'] = ts_results
        
        return ts_results
    
    def train_classification_models(self):
        """Train classification models to predict categories"""
        print("\n=== Training Classification Models ===")
        
        features_df = self.processed_data['features_df']
        
        # Create target variable: classify production level (low, medium, high)
        features_df = features_df.dropna(subset=['mean_production'])
        production_thresholds = np.percentile(features_df['mean_production'], [33, 66])
        
        def classify_production(value):
            if value <= production_thresholds[0]:
                return 0  # Low
            elif value <= production_thresholds[1]:
                return 1  # Medium
            else:
                return 2  # High
        
        features_df['production_class'] = features_df['mean_production'].apply(classify_production)
        
        # Prepare features and target
        feature_cols = ['Area_encoded', 'Item_encoded', 'std_production', 'min_production', 'max_production']
        X = features_df[feature_cols]
        y = features_df['production_class']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Define classification models
        clf_models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest Classifier': RandomForestClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(kernel='rbf', random_state=42, probability=True),
            'XGBoost Classifier': xgb.XGBClassifier(random_state=42)
        }
        
        clf_results = {}
        
        for name, model in clf_models.items():
            print(f"Training {name}...")
            
            # Fit model
            if name in ['SVM']:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                y_proba = model.predict_proba(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            
            # Cross-validation
            if name in ['SVM']:
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
            else:
                cv_scores = cross_val_score(model, X_train, y_train, cv=5)
            
            clf_results[name] = {
                'accuracy': accuracy,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'predictions': y_pred.tolist(),
                'actual': y_test.tolist(),
                'probabilities': y_proba.tolist(),
                'classification_report': classification_report(y_test, y_pred, output_dict=True),
                'model': model
            }
            
            print(f"  Accuracy: {accuracy:.4f}, CV: {cv_scores.mean():.4f}±{cv_scores.std():.4f}")
        
        # Store results
        self.models['classification'] = clf_results
        self.results['classification'] = clf_results
        
        # Save scaler
        import joblib
        joblib.dump(scaler, os.path.join(self.model_dir, 'classification_scaler.pkl'))
        
        return clf_results
    
    def create_clustering_models(self):
        """Create clustering models for agricultural patterns"""
        print("\n=== Creating Clustering Models ===")
        
        features_df = self.processed_data['features_df']
        
        # Use numerical features for clustering
        feature_cols = ['mean_production', 'std_production', 'min_production', 'max_production']
        X = features_df[feature_cols].dropna()
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply different clustering algorithms
        cluster_results = {}
        
        # K-Means clustering
        kmeans = KMeans(n_clusters=5, random_state=42)
        kmeans_labels = kmeans.fit_predict(X_scaled)
        
        # PCA for dimensionality reduction and visualization
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        cluster_results['kmeans'] = {
            'labels': kmeans_labels.tolist(),
            'centroids': kmeans.cluster_centers_.tolist(),
            'inertia': kmeans.inertia_,
            'X_pca': X_pca.tolist(),
            'model': kmeans,
            'pca_model': pca
        }
        
        print(f"K-Means clusters created with inertia: {kmeans.inertia_:.4f}")
        
        # Store results
        self.models['clustering'] = cluster_results
        self.results['clustering'] = cluster_results
        
        # Save models
        import joblib
        joblib.dump(kmeans, os.path.join(self.model_dir, 'kmeans_model.pkl'))
        joblib.dump(pca, os.path.join(self.model_dir, 'pca_model.pkl'))
        joblib.dump(scaler, os.path.join(self.model_dir, 'clustering_scaler.pkl'))
        
        return cluster_results
    
    def create_ensemble_models(self):
        """Create ensemble models combining approaches"""
        print("\n=== Creating Ensemble Models ===")
        
        ensemble_results = {}
        
        # If we have multiple regression models, create a simple ensemble
        if 'regression' in self.results:
            reg_results = self.results['regression']
            
            # Find the best regression models
            best_models = sorted(reg_results.items(), 
                               key=lambda x: x[1]['r2_score'], reverse=True)[:3]
            
            print(f"Top 3 regression models: {[name for name, _ in best_models]}")
            
            ensemble_results['regression_ensemble'] = {
                'top_models': [name for name, _ in best_models],
                'combined_metrics': {
                    'best_r2': best_models[0][1]['r2_score'],
                    'best_rmse': best_models[0][1]['rmse']
                }
            }
        
        # If we have multiple classification models
        if 'classification' in self.results:
            clf_results = self.results['classification']
            
            # Find the best classification models
            best_clf_models = sorted(clf_results.items(), 
                                   key=lambda x: x[1]['accuracy'], reverse=True)[:3]
            
            print(f"Top 3 classification models: {[name for name, _ in best_clf_models]}")
            
            ensemble_results['classification_ensemble'] = {
                'top_models': [name for name, _ in best_clf_models],
                'combined_metrics': {
                    'best_accuracy': best_clf_models[0][1]['accuracy']
                }
            }
        
        self.results['ensemble'] = ensemble_results
        self.models['ensemble'] = ensemble_results
        
        return ensemble_results
    
    def generate_reports(self):
        """Generate comprehensive reports"""
        print("\n=== Generating Reports ===")
        
        # Overall summary
        summary = {
            'timestamp': self.timestamp,
            'dataset_info': {
                'original_shape': self.df.shape if self.df is not None else None,
                'processed_features': len(self.processed_data.get('features_df', [])),
            },
            'models_trained': {},
            'best_performers': {}
        }
        
        # Add regression results
        if 'regression' in self.results:
            reg_results = self.results['regression']
            summary['models_trained']['regression'] = list(reg_results.keys())
            
            best_reg = max(reg_results.items(), key=lambda x: x[1]['r2_score'])
            summary['best_performers']['regression'] = {
                'model': best_reg[0],
                'r2_score': best_reg[1]['r2_score'],
                'rmse': best_reg[1]['rmse']
            }
        
        # Add classification results
        if 'classification' in self.results:
            clf_results = self.results['classification']
            summary['models_trained']['classification'] = list(clf_results.keys())
            
            best_clf = max(clf_results.items(), key=lambda x: x[1]['accuracy'])
            summary['best_performers']['classification'] = {
                'model': best_clf[0],
                'accuracy': best_clf[1]['accuracy']
            }
        
        # Save summary
        summary_path = os.path.join(self.report_dir, "agriculture_ml_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Create detailed reports for each model type
        if 'regression' in self.results:
            reg_report = "# Regression Models Report\n\n"
            for name, result in self.results['regression'].items():
                reg_report += f"## {name}\n"
                reg_report += f"- R² Score: {result['r2_score']:.4f}\n"
                reg_report += f"- RMSE: {result['rmse']:.4f}\n"
                reg_report += f"- Cross-validation R²: {result['cv_mean']:.4f}±{result['cv_std']:.4f}\n\n"
            
            with open(os.path.join(self.report_dir, "regression_report.txt"), 'w') as f:
                f.write(reg_report)
        
        if 'classification' in self.results:
            clf_report = "# Classification Models Report\n\n"
            for name, result in self.results['classification'].items():
                clf_report += f"## {name}\n"
                clf_report += f"- Accuracy: {result['accuracy']:.4f}\n"
                clf_report += f"- Cross-validation: {result['cv_mean']:.4f}±{result['cv_std']:.4f}\n\n"
            
            with open(os.path.join(self.report_dir, "classification_report.txt"), 'w') as f:
                f.write(clf_report)
        
        print(f"Reports saved to: {self.report_dir}")
        return summary
    
    def generate_visualizations(self):
        """Generate comprehensive visualizations"""
        print("\n=== Generating Visualizations ===")
        
        # 1. Model Performance Comparison - Regression
        if 'regression' in self.results:
            plt.figure(figsize=(12, 8))
            
            models = list(self.results['regression'].keys())
            r2_scores = [self.results['regression'][m]['r2_score'] for m in models]
            rmse_scores = [self.results['regression'][m]['rmse'] for m in models]
            
            # Plot R² scores
            plt.subplot(2, 2, 1)
            bars1 = plt.bar(models, r2_scores, color='skyblue', alpha=0.7)
            plt.title('Regression Models - R² Score Comparison')
            plt.ylabel('R² Score')
            plt.xticks(rotation=45, ha='right')
            
            # Add value labels
            for bar, score in zip(bars1, r2_scores):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{score:.3f}', ha='center', va='bottom', fontsize=8)
            
            # Plot RMSE scores
            plt.subplot(2, 2, 2)
            bars2 = plt.bar(models, rmse_scores, color='lightcoral', alpha=0.7)
            plt.title('Regression Models - RMSE Comparison')
            plt.ylabel('RMSE')
            plt.xticks(rotation=45, ha='right')
            
            # Add value labels
            for bar, score in zip(bars2, rmse_scores):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{score:.3f}', ha='center', va='bottom', fontsize=8)
        
        # 2. Model Performance Comparison - Classification
        if 'classification' in self.results:
            plt.subplot(2, 2, 3)
            clf_models = list(self.results['classification'].keys())
            clf_accuracies = [self.results['classification'][m]['accuracy'] for m in clf_models]
            
            bars3 = plt.bar(clf_models, clf_accuracies, color='lightgreen', alpha=0.7)
            plt.title('Classification Models - Accuracy Comparison')
            plt.ylabel('Accuracy')
            plt.xticks(rotation=45, ha='right')
            
            # Add value labels
            for bar, score in zip(bars3, clf_accuracies):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{score:.3f}', ha='center', va='bottom', fontsize=8)
        
        # 3. Clustering Visualization (if available)
        if 'clustering' in self.results:
            plt.subplot(2, 2, 4)
            cluster_data = self.results['clustering']['kmeans']
            X_pca = np.array(cluster_data['X_pca'])
            labels = np.array(cluster_data['labels'])
            
            scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', alpha=0.6)
            plt.title('Clustering Results (PCA)')
            plt.xlabel('Principal Component 1')
            plt.ylabel('Principal Component 2')
            plt.colorbar(scatter)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.vis_dir, "model_performance_comparison.png"), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Additional visualizations
        if 'regression' in self.results:
            # Best regression model predictions vs actual
            best_reg = max(self.results['regression'].items(), 
                          key=lambda x: x[1]['r2_score'])
            
            plt.figure(figsize=(10, 6))
            actual = best_reg[1]['actual']
            pred = best_reg[1]['predictions']
            
            plt.scatter(actual, pred, alpha=0.6)
            plt.plot([min(actual), max(actual)], [min(actual), max(actual)], 'r--', lw=2)
            plt.xlabel('Actual Values')
            plt.ylabel('Predicted Values')
            plt.title(f'Best Regression Model ({best_reg[0]}) - Predictions vs Actual')
            plt.tight_layout()
            plt.savefig(os.path.join(self.vis_dir, "regression_predictions.png"), 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"Visualizations saved to: {self.vis_dir}")
    
    def run_complete_pipeline(self):
        """Run the complete agriculture ML pipeline"""
        print("=" * 70)
        print("AGRICULTURE DATASET ML PIPELINE - ALL POSSIBLE APPROACHES")
        print("=" * 70)
        
        # Load and explore data
        self.load_and_explore_data()
        
        # Preprocess data
        self.preprocess_data()
        
        # Train models
        self.train_regression_models()
        self.train_time_series_forecasting()
        self.train_classification_models()
        self.create_clustering_models()
        self.create_ensemble_models()
        
        # Generate reports and visualizations
        self.generate_reports()
        self.generate_visualizations()
        
        # Print summary
        print("\n" + "=" * 70)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print(f"Models saved to: {self.model_dir}")
        print(f"Reports saved to: {self.report_dir}")
        print(f"Visualizations saved to: {self.vis_dir}")
        
        # Best performers
        if 'regression' in self.results:
            best_reg = max(self.results['regression'].items(), 
                          key=lambda x: x[1]['r2_score'])
            print(f"\nBest Regression Model: {best_reg[0]} (R² = {best_reg[1]['r2_score']:.4f})")
        
        if 'classification' in self.results:
            best_clf = max(self.results['classification'].items(), 
                          key=lambda x: x[1]['accuracy'])
            print(f"Best Classification Model: {best_clf[0]} (Accuracy = {best_clf[1]['accuracy']:.4f})")
        
        return self.results

if __name__ == "__main__":
    # Run the complete pipeline
    pipeline = AgricultureMLPipeline()
    results = pipeline.run_complete_pipeline()