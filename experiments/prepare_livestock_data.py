"""
Comprehensive Livestock Census Data Analysis and ML Training
Analyzes all livestock datasets and trains models in every possible way
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import xgboost as xgb
import lightgbm as lgb
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import warnings
warnings.filterwarnings('ignore')
import glob
from datetime import datetime
import json

class LivestockCensusAnalyzer:
    """Comprehensive analyzer for livestock census data"""
    
    def __init__(self, data_dir="data/livestock_census"):
        self.data_dir = data_dir
        self.datasets = {}
        self.combined_data = None
        self.models = {}
        self.results = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create output directories
        self.model_dir = f"models/livestock_census_{self.timestamp}"
        self.report_dir = f"reports/livestock_census_{self.timestamp}"
        self.vis_dir = f"visualizations/livestock_census_{self.timestamp}"
        
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.report_dir, exist_ok=True)
        os.makedirs(self.vis_dir, exist_ok=True)
    
    def load_all_datasets(self):
        """Load all livestock census datasets"""
        print("Loading all livestock census datasets...")
        
        # Get all CSV files in the livestock census directory
        csv_files = glob.glob(os.path.join(self.data_dir, "*.csv"))
        
        if not csv_files:
            print(f"No CSV files found in {self.data_dir}")
            return
        
        for file_path in csv_files:
            file_name = os.path.basename(file_path)
            print(f"Loading: {file_name}")
            
            try:
                # Read the CSV file
                df = pd.read_csv(file_path)
                
                # Store the dataset
                self.datasets[file_name] = {
                    'data': df,
                    'state': self._extract_state_from_filename(file_name),
                    'table_type': self._extract_table_type(file_name)
                }
                
                print(f"  - Shape: {df.shape}")
                print(f"  - Columns: {list(df.columns)}")
                
            except Exception as e:
                print(f"  - Error loading {file_name}: {e}")
        
        print(f"\nLoaded {len(self.datasets)} datasets successfully")
        return self.datasets
    
    def _extract_state_from_filename(self, filename):
        """Extract state name from filename"""
        parts = filename.split('-')
        if len(parts) > 0:
            return parts[0]
        return "Unknown"
    
    def _extract_table_type(self, filename):
        """Extract table type from filename"""
        if 'Table2A' in filename:
            return 'Exotic_Crossbred'
        elif 'Table2B' in filename:
            return 'Indigenous'
        else:
            return 'General'
    
    def explore_data(self):
        """Explore and understand the livestock data"""
        print("\n" + "="*60)
        print("LIVESTOCK DATA EXPLORATION")
        print("="*60)
        
        # Combine all datasets for overall analysis
        all_data_list = []
        
        for filename, dataset_info in self.datasets.items():
            df = dataset_info['data']
            
            # Add state and table type as columns
            df_copy = df.copy()
            df_copy['State'] = dataset_info['state']
            df_copy['Table_Type'] = dataset_info['table_type']
            
            all_data_list.append(df_copy)
        
        if all_data_list:
            self.combined_data = pd.concat(all_data_list, ignore_index=True, sort=False)
            print(f"Combined dataset shape: {self.combined_data.shape}")
            print(f"Number of unique states: {self.combined_data['State'].nunique()}")
            print(f"Table types present: {self.combined_data['Table_Type'].unique()}")
            
            # Display basic info
            print(f"\nColumn names in combined dataset:")
            for i, col in enumerate(self.combined_data.columns):
                print(f"  {i+1}. {col}")
        
        return self.combined_data
    
    def preprocess_data(self):
        """Preprocess the livestock data for different modeling approaches"""
        print("\n" + "="*60)
        print("DATA PREPROCESSING")
        print("="*60)
        
        if self.combined_data is None:
            print("No data to preprocess. Run explore_data() first.")
            return
        
        # Handle different column names for districts
        district_col_options = ['Districts', 'District Name', 'district', 'District']
        district_col = None
        for col in self.combined_data.columns:
            if col in district_col_options:
                district_col = col
                break
        
        if district_col:
            print(f"Using '{district_col}' as district column")
        else:
            print("No district column found, using index")
            district_col = 'Index'
            self.combined_data['Index'] = self.combined_data.index
        
        # Create features based on livestock data
        # Look for columns related to cattle/breeds
        cattle_related_cols = []
        for col in self.combined_data.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['cattle', 'total', 'breed', 'indigenous', 'exotic', 'crossbred']):
                cattle_related_cols.append(col)
        
        print(f"Cattle-related columns found: {len(cattle_related_cols)}")
        for col in cattle_related_cols:
            print(f"  - {col}")
        
        # Prepare data for different modeling tasks
        self.processed_data = {
            'regression_features': [],
            'classification_features': [],
            'clustering_features': [],
            'time_series_features': []  # Though we only have 2007 data
        }
        
        # Identify numeric columns for modeling
        numeric_cols = self.combined_data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Filter out columns that have meaningful livestock data
        livestock_cols = []
        for col in numeric_cols:
            # Skip state and table type encoded columns if they exist
            if col not in ['State', 'Table_Type']:
                livestock_cols.append(col)
        
        print(f"Numeric livestock columns for analysis: {len(livestock_cols)}")
        
        # Prepare different datasets for different ML tasks
        self.livestock_data = self.combined_data[livestock_cols + ['State', 'Table_Type']].copy()
        
        # Remove rows with all NaN in livestock columns
        self.livestock_data = self.livestock_data.dropna(subset=livestock_cols, how='all')
        
        print(f"After cleaning, livestock data shape: {self.livestock_data.shape}")
        
        return self.livestock_data
    
    def train_regression_models(self):
        """Train regression models to predict livestock numbers"""
        print("\n" + "="*60)
        print("REGRESSION MODEL TRAINING")
        print("="*60)
        
        if self.livestock_data is None:
            print("No data available for regression. Run preprocess_data() first.")
            return
        
        # Identify potential target columns (usually totals)
        numeric_cols = self.livestock_data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Look for columns that might be totals
        potential_targets = []
        for col in numeric_cols:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['total', 'total']):
                potential_targets.append(col)
        
        # If no total columns found, use one of the largest columns as target
        if not potential_targets:
            # Find column with highest average values
            avg_values = self.livestock_data[numeric_cols].mean().sort_values(ascending=False)
            potential_targets = avg_values.head(3).index.tolist()
        
        print(f"Potential target columns: {potential_targets}")
        
        # Use the first potential target as our main target
        if potential_targets:
            target_col = potential_targets[0]
            print(f"Selected target column: {target_col}")
            
            # Prepare features (exclude target and categorical columns)
            feature_cols = [col for col in numeric_cols if col != target_col]
            
            # Remove rows where target is NaN
            data_for_modeling = self.livestock_data[[target_col] + feature_cols].dropna(subset=[target_col])
            
            if len(data_for_modeling) > 10:  # Need sufficient data
                X = data_for_modeling[feature_cols]
                y = data_for_modeling[target_col]
                
                # Fill remaining NaN values
                X = X.fillna(X.mean())
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Define regression models
                regression_models = {
                    'Linear Regression': LinearRegression(),
                    'Random Forest Regressor': RandomForestRegressor(n_estimators=100, random_state=42),
                    'Gradient Boosting Regressor': GradientBoostingRegressor(n_estimators=100, random_state=42),
                    'Decision Tree Regressor': DecisionTreeRegressor(random_state=42),
                    'K-Nearest Neighbors': KNeighborsRegressor(n_neighbors=5),
                    'XGBoost Regressor': xgb.XGBRegressor(random_state=42),
                    'LightGBM Regressor': lgb.LGBMRegressor(random_state=42)
                }
                
                reg_results = {}
                
                for name, model in regression_models.items():
                    print(f"Training {name}...")
                    
                    try:
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
                        
                    except Exception as e:
                        print(f"  Error training {name}: {e}")
                
                # Store results
                self.models['regression'] = reg_results
                self.results['regression'] = reg_results
                
                # Save scaler
                import joblib
                joblib.dump(scaler, os.path.join(self.model_dir, 'regression_scaler.pkl'))
                
                return reg_results
            else:
                print("Not enough data for regression modeling after cleaning")
        
        return {}
    
    def train_classification_models(self):
        """Train classification models for livestock categories"""
        print("\n" + "="*60)
        print("CLASSIFICATION MODEL TRAINING")
        print("="*60)
        
        if self.livestock_data is None:
            print("No data available for classification. Run preprocess_data() first.")
            return
        
        # Create classification tasks
        # We'll create categories based on livestock numbers
        numeric_cols = self.livestock_data.select_dtypes(include=[np.number]).columns.tolist()
        
        if numeric_cols:
            # Use the first numeric column as base for classification
            base_col = numeric_cols[0]
            
            # Remove rows with NaN in base column
            data_for_class = self.livestock_data[[base_col] + numeric_cols].dropna(subset=[base_col])
            
            if len(data_for_class) > 10:
                # Create target categories based on percentiles
                base_values = data_for_class[base_col]
                if len(base_values) >= 3:  # Ensure we have enough data points
                    percentiles_vals = np.percentile(base_values, [33, 66])
                    
                    # Create categorical target using numpy.where for vectorized operation
                    y_cat = np.where(base_values <= percentiles_vals[0], 0,  # Low
                                     np.where(base_values <= percentiles_vals[1], 1,  # Medium
                                              2))  # High
                    
                    # Ensure y_cat is a 1D array
                    y_cat = y_cat.flatten() if hasattr(y_cat, 'flatten') else np.array(y_cat)
                    
                    # Use other columns as features
                    feature_cols = [col for col in numeric_cols if col != base_col]
                    X = data_for_class[feature_cols].fillna(0)  # Fill with 0 for classification
                    
                    # Check if X and y_cat have the same number of samples
                    if len(X) != len(y_cat):
                        print(f"Mismatch in sample counts: X={len(X)}, y_cat={len(y_cat)}")
                        print("Using only the common samples")
                        min_samples = min(len(X), len(y_cat))
                        X = X.iloc[:min_samples]
                        y_cat = y_cat[:min_samples]
                else:
                    print("Not enough data points for classification after cleaning")
                    return {}
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42)
                
                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Define classification models
                classification_models = {
                    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
                    'Random Forest Classifier': RandomForestClassifier(n_estimators=100, random_state=42),
                    'SVM': SVC(random_state=42),
                    'Decision Tree Classifier': DecisionTreeClassifier(random_state=42),
                    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
                    'XGBoost Classifier': xgb.XGBClassifier(random_state=42),
                    'Naive Bayes': GaussianNB()
                }
                
                clf_results = {}
                
                for name, model in classification_models.items():
                    print(f"Training {name}...")
                    
                    try:
                        # Fit model
                        if name in ['SVM', 'Naive Bayes']:
                            model.fit(X_train_scaled, y_train)
                            y_pred = model.predict(X_test_scaled)
                        else:
                            model.fit(X_train, y_train)
                            y_pred = model.predict(X_test)
                        
                        # Calculate metrics
                        accuracy = accuracy_score(y_test, y_pred)
                        
                        # Cross-validation
                        if name in ['SVM', 'Naive Bayes']:
                            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
                        else:
                            cv_scores = cross_val_score(model, X_train, y_train, cv=5)
                        
                        clf_results[name] = {
                            'accuracy': accuracy,
                            'cv_mean': cv_scores.mean(),
                            'cv_std': cv_scores.std(),
                            'predictions': y_pred.tolist(),
                            'actual': y_test.tolist(),
                            'classification_report': classification_report(y_test, y_pred, output_dict=True),
                            'model': model
                        }
                        
                        print(f"  Accuracy: {accuracy:.4f}, CV: {cv_scores.mean():.4f}±{cv_scores.std():.4f}")
                        
                    except Exception as e:
                        print(f"  Error training {name}: {e}")
                
                # Store results
                self.models['classification'] = clf_results
                self.results['classification'] = clf_results
                
                # Save scaler
                import joblib
                joblib.dump(scaler, os.path.join(self.model_dir, 'classification_scaler.pkl'))
                
                return clf_results
            else:
                print("Not enough data for classification modeling")
                return {}
        else:
            print("No numeric columns available for classification")
            return {}
    
    def create_clustering_models(self):
        """Create clustering models for livestock patterns"""
        print("\n" + "="*60)
        print("CLUSTERING MODEL TRAINING")
        print("="*60)
        
        if self.livestock_data is None:
            print("No data available for clustering. Run preprocess_data() first.")
            return
        
        # Use numeric columns for clustering
        numeric_cols = self.livestock_data.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) > 1:
            # Prepare data
            X = self.livestock_data[numeric_cols].copy()
            X = X.fillna(X.mean())  # Fill missing values
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Apply clustering algorithms
            cluster_results = {}
            
            # K-Means clustering with different numbers of clusters
            kmeans_results = {}
            for n_clusters in [3, 4, 5, 6]:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                labels = kmeans.fit_predict(X_scaled)
                
                kmeans_results[f'KMeans_{n_clusters}'] = {
                    'labels': labels.tolist(),
                    'centroids': kmeans.cluster_centers_.tolist(),
                    'inertia': kmeans.inertia_,
                    'model': kmeans
                }
            
            cluster_results['kmeans'] = kmeans_results
            
            # PCA for dimensionality reduction and visualization
            pca = PCA(n_components=min(2, X_scaled.shape[1]))
            X_pca = pca.fit_transform(X_scaled)
            
            cluster_results['pca'] = {
                'components': X_pca.tolist(),
                'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
                'pca_model': pca
            }
            
            print(f"Created clustering models:")
            for model_name in kmeans_results.keys():
                print(f"  - {model_name} with inertia: {kmeans_results[model_name]['inertia']:.4f}")
            print(f"  - PCA with {len(pca.components_)} components explaining {sum(pca.explained_variance_ratio_):.4f} variance")
            
            # Store results
            self.models['clustering'] = cluster_results
            self.results['clustering'] = cluster_results
            
            # Save models
            import joblib
            joblib.dump(scaler, os.path.join(self.model_dir, 'clustering_scaler.pkl'))
            joblib.dump(pca, os.path.join(self.model_dir, 'pca_model.pkl'))
            
            return cluster_results
        else:
            print("Not enough numeric columns for clustering")
            return {}
    
    def generate_reports(self):
        """Generate comprehensive reports"""
        print("\n" + "="*60)
        print("GENERATING REPORTS")
        print("="*60)
        
        # Overall summary
        summary = {
            'timestamp': self.timestamp,
            'dataset_info': {
                'total_datasets_loaded': len(self.datasets),
                'combined_data_shape': self.combined_data.shape if self.combined_data is not None else None,
                'states_represented': self.combined_data['State'].nunique() if self.combined_data is not None else 0,
            },
            'models_trained': {},
            'best_performers': {}
        }
        
        # Add regression results if available
        if 'regression' in self.results:
            reg_results = self.results['regression']
            summary['models_trained']['regression'] = list(reg_results.keys())
            
            if reg_results:
                best_reg = max(reg_results.items(), key=lambda x: x[1]['r2_score'])
                summary['best_performers']['regression'] = {
                    'model': best_reg[0],
                    'r2_score': best_reg[1]['r2_score'],
                    'rmse': best_reg[1]['rmse']
                }
        
        # Add classification results if available
        if 'classification' in self.results:
            clf_results = self.results['classification']
            summary['models_trained']['classification'] = list(clf_results.keys())
            
            if clf_results:
                best_clf = max(clf_results.items(), key=lambda x: x[1]['accuracy'])
                summary['best_performers']['classification'] = {
                    'model': best_clf[0],
                    'accuracy': best_clf[1]['accuracy']
                }
        
        # Save summary
        summary_path = os.path.join(self.report_dir, "livestock_census_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Create detailed reports
        report_content = f"""LIVESTOCK CENSUS ANALYSIS REPORT
Generated on: {self.timestamp}

DATASET OVERVIEW:
- Total datasets loaded: {len(self.datasets)}
- Combined data shape: {self.combined_data.shape if self.combined_data is not None else 'N/A'}
- States represented: {self.combined_data['State'].nunique() if self.combined_data is not None else 0}
- Table types: {list(self.combined_data['Table_Type'].unique()) if self.combined_data is not None else 'N/A'}

MODEL PERFORMANCE:
"""
        
        if 'regression' in self.results:
            report_content += "\nREGRESSION MODELS:\n"
            for name, result in self.results['regression'].items():
                report_content += f"- {name}: R² = {result['r2_score']:.4f}, RMSE = {result['rmse']:.4f}\n"
        
        if 'classification' in self.results:
            report_content += "\nCLASSIFICATION MODELS:\n"
            for name, result in self.results['classification'].items():
                report_content += f"- {name}: Accuracy = {result['accuracy']:.4f}\n"
        
        if 'clustering' in self.results:
            report_content += "\nCLUSTERING RESULTS:\n"
            if 'kmeans' in self.results['clustering']:
                for model_name, result in self.results['clustering']['kmeans'].items():
                    report_content += f"- {model_name}: Inertia = {result['inertia']:.4f}\n"
        
        report_path = os.path.join(self.report_dir, "livestock_census_analysis_report.txt")
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        print(f"Reports saved to: {self.report_dir}")
        return summary
    
    def generate_visualizations(self):
        """Generate comprehensive visualizations"""
        print("\n" + "="*60)
        print("GENERATING VISUALIZATIONS")
        print("="*60)
        
        # Create visualizations if we have data
        if self.combined_data is not None:
            plt.style.use('default')
            
            # 1. State distribution
            if 'State' in self.combined_data.columns:
                plt.figure(figsize=(12, 6))
                state_counts = self.combined_data['State'].value_counts()
                plt.bar(range(len(state_counts)), state_counts.values)
                plt.title('Distribution of Data Across States')
                plt.xlabel('States (Index)')
                plt.ylabel('Number of Records')
                plt.xticks(range(len(state_counts)), state_counts.index, rotation=45)
                plt.tight_layout()
                plt.savefig(os.path.join(self.vis_dir, "state_distribution.png"), dpi=300, bbox_inches='tight')
                plt.close()
            
            # 2. Model performance comparison (if models were trained)
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Regression model comparison
            if 'regression' in self.results and self.results['regression']:
                reg_results = self.results['regression']
                models = list(reg_results.keys())
                r2_scores = [reg_results[m]['r2_score'] for m in models]
                rmse_scores = [reg_results[m]['rmse'] for m in models]
                
                axes[0, 0].bar(models, r2_scores)
                axes[0, 0].set_title('Regression Models - R² Score')
                axes[0, 0].set_ylabel('R² Score')
                axes[0, 0].tick_params(axis='x', rotation=45)
                
                axes[0, 1].bar(models, rmse_scores)
                axes[0, 1].set_title('Regression Models - RMSE')
                axes[0, 1].set_ylabel('RMSE')
                axes[0, 1].tick_params(axis='x', rotation=45)
            
            # Classification model comparison
            if 'classification' in self.results and self.results['classification']:
                clf_results = self.results['classification']
                models = list(clf_results.keys())
                accuracies = [clf_results[m]['accuracy'] for m in models]
                
                axes[1, 0].bar(models, accuracies)
                axes[1, 0].set_title('Classification Models - Accuracy')
                axes[1, 0].set_ylabel('Accuracy')
                axes[1, 0].tick_params(axis='x', rotation=45)
            
            # Clustering visualization (if PCA is available)
            if 'clustering' in self.results and 'pca' in self.results['clustering']:
                pca_data = np.array(self.results['clustering']['pca']['components'])
                if pca_data.shape[1] >= 2:
                    scatter = axes[1, 1].scatter(pca_data[:, 0], pca_data[:, 1], alpha=0.6)
                    axes[1, 1].set_title('PCA Visualization of Livestock Data')
                    axes[1, 1].set_xlabel('Principal Component 1')
                    axes[1, 1].set_ylabel('Principal Component 2')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.vis_dir, "model_performance_comparison.png"), 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"Visualizations saved to: {self.vis_dir}")
    
    def run_complete_analysis(self):
        """Run the complete livestock census analysis"""
        print("="*70)
        print("LIVESTOCK CENSUS COMPLETE ANALYSIS PIPELINE")
        print("Training models in EVERY POSSIBLE WAY")
        print("="*70)
        
        # Load all datasets
        self.load_all_datasets()
        
        # Explore data
        self.explore_data()
        
        # Preprocess data
        self.preprocess_data()
        
        # Train models
        self.train_regression_models()
        self.train_classification_models()
        self.create_clustering_models()
        
        # Generate reports and visualizations
        self.generate_reports()
        self.generate_visualizations()
        
        # Print summary
        print("\n" + "="*70)
        print("ANALYSIS COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f"Models saved to: {self.model_dir}")
        print(f"Reports saved to: {self.report_dir}")
        print(f"Visualizations saved to: {self.vis_dir}")
        
        # Performance summary
        if 'regression' in self.results:
            print(f"\nBest Regression Model: {max(self.results['regression'].items(), key=lambda x: x[1]['r2_score'])[0] if self.results['regression'] else 'None'}")
        if 'classification' in self.results:
            print(f"Best Classification Model: {max(self.results['classification'].items(), key=lambda x: x[1]['accuracy'])[0] if self.results['classification'] else 'None'}")
        
        return self.results

if __name__ == "__main__":
    # Run the complete analysis
    analyzer = LivestockCensusAnalyzer()
    results = analyzer.run_complete_analysis()