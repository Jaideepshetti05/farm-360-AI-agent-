"""
Comprehensive Animal Disease Prediction Models
Trains models in every possible way for animal disease detection
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import xgboost as xgb
import lightgbm as lgb
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime
import joblib
import json

class AnimalDiseasePredictor:
    """Comprehensive predictor for animal diseases using multiple ML approaches"""
    
    def __init__(self, data_path="health_module/animal_disease_dataset.csv"):
        self.data_path = data_path
        self.df = None
        self.models = {}
        self.results = {}
        self.preprocessors = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create output directories
        self.model_dir = f"models/animal_disease_{self.timestamp}"
        self.report_dir = f"reports/animal_disease_{self.timestamp}"
        self.vis_dir = f"visualizations/animal_disease_{self.timestamp}"
        
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.report_dir, exist_ok=True)
        os.makedirs(self.vis_dir, exist_ok=True)
        
    def load_data(self):
        """Load the animal disease dataset"""
        print("Loading animal disease dataset...")
        self.df = pd.read_csv(self.data_path)
        print(f"Dataset shape: {self.df.shape}")
        print(f"Columns: {self.df.columns.tolist()}")
        print(f"Disease distribution:\n{self.df['Disease'].value_counts()}")
        return self.df
    
    def preprocess_data(self):
        """Preprocess the data for modeling"""
        print("\n" + "="*60)
        print("DATA PREPROCESSING")
        print("="*60)
        
        # Encode categorical variables
        self.label_encoders = {}
        df_processed = self.df.copy()
        
        categorical_columns = ['Animal', 'Symptom 1', 'Symptom 2', 'Symptom 3', 'Disease']
        
        for col in categorical_columns:
            le = LabelEncoder()
            df_processed[col + '_encoded'] = le.fit_transform(df_processed[col].astype(str))
            self.label_encoders[col] = le
        
        # Prepare features and target
        feature_columns = ['Animal_encoded', 'Age', 'Temperature', 'Symptom 1_encoded', 'Symptom 2_encoded', 'Symptom 3_encoded']
        self.X = df_processed[feature_columns]
        self.y = df_processed['Disease_encoded']
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        
        print(f"Training set shape: {self.X_train.shape}")
        print(f"Test set shape: {self.X_test.shape}")
        print(f"Feature columns: {feature_columns}")
        
        # Scale features
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        self.preprocessors['scaler'] = self.scaler
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_classification_models(self):
        """Train multiple classification models"""
        print("\n" + "="*60)
        print("CLASSIFICATION MODEL TRAINING")
        print("="*60)
        
        # Define classification models
        classification_models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest Classifier': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting Classifier': GradientBoostingClassifier(random_state=42),
            'SVM': SVC(random_state=42, probability=True),
            'Decision Tree Classifier': DecisionTreeClassifier(random_state=42),
            'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
            'XGBoost Classifier': xgb.XGBClassifier(random_state=42),
            'LightGBM Classifier': lgb.LGBMClassifier(random_state=42),
            'AdaBoost Classifier': AdaBoostClassifier(random_state=42),
            'Extra Trees Classifier': ExtraTreesClassifier(n_estimators=100, random_state=42),
            'Naive Bayes': GaussianNB()
        }
        
        results = {}
        
        for name, model in classification_models.items():
            print(f"Training {name}...")
            
            try:
                # Fit model
                if name in ['SVM', 'Naive Bayes', 'Logistic Regression', 'K-Nearest Neighbors']:
                    model.fit(self.X_train_scaled, self.y_train)
                    y_pred = model.predict(self.X_test_scaled)
                    y_pred_proba = model.predict_proba(self.X_test_scaled) if hasattr(model, 'predict_proba') else None
                else:
                    model.fit(self.X_train, self.y_train)
                    y_pred = model.predict(self.X_test)
                    y_pred_proba = model.predict_proba(self.X_test) if hasattr(model, 'predict_proba') else None
                
                # Calculate metrics
                accuracy = accuracy_score(self.y_test, y_pred)
                f1 = f1_score(self.y_test, y_pred, average='weighted')
                
                # Cross-validation
                if name in ['SVM', 'Naive Bayes', 'Logistic Regression', 'K-Nearest Neighbors']:
                    cv_scores = cross_val_score(model, self.X_train_scaled, self.y_train, cv=5, scoring='accuracy')
                else:
                    cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5, scoring='accuracy')
                
                results[name] = {
                    'accuracy': accuracy,
                    'f1_score': f1,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'predictions': y_pred.tolist(),
                    'actual': self.y_test.tolist(),
                    'classification_report': classification_report(self.y_test, y_pred, output_dict=True),
                    'confusion_matrix': confusion_matrix(self.y_test, y_pred).tolist(),
                    'model': model,
                    'prediction_probabilities': y_pred_proba.tolist() if y_pred_proba is not None else None
                }
                
                print(f"  Accuracy: {accuracy:.4f}, F1: {f1:.4f}, CV: {cv_scores.mean():.4f}±{cv_scores.std():.4f}")
                
            except Exception as e:
                print(f"  Error training {name}: {e}")
        
        # Store results
        self.models['classification'] = results
        self.results['classification'] = results
        
        # Save scaler
        joblib.dump(self.scaler, os.path.join(self.model_dir, 'classification_scaler.pkl'))
        
        return results
    
    def create_ensemble_models(self):
        """Create ensemble models using voting classifier"""
        print("\n" + "="*60)
        print("ENSEMBLE MODEL TRAINING")
        print("="*60)
        
        # Select top performing models for ensemble
        base_models = [
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
            ('xgb', xgb.XGBClassifier(random_state=42)),
            ('lgb', lgb.LGBMClassifier(random_state=42)),
            ('svm', SVC(random_state=42, probability=True))
        ]
        
        # Soft voting classifier
        voting_clf_soft = VotingClassifier(estimators=base_models, voting='soft')
        voting_clf_soft.fit(self.X_train, self.y_train)
        y_pred_soft = voting_clf_soft.predict(self.X_test)
        
        # Hard voting classifier
        voting_clf_hard = VotingClassifier(estimators=base_models, voting='hard')
        voting_clf_hard.fit(self.X_train, self.y_train)
        y_pred_hard = voting_clf_hard.predict(self.X_test)
        
        soft_accuracy = accuracy_score(self.y_test, y_pred_soft)
        hard_accuracy = accuracy_score(self.y_test, y_pred_hard)
        
        ensemble_results = {
            'Soft Voting': {
                'accuracy': soft_accuracy,
                'predictions': y_pred_soft.tolist(),
                'actual': self.y_test.tolist(),
                'model': voting_clf_soft
            },
            'Hard Voting': {
                'accuracy': hard_accuracy,
                'predictions': y_pred_hard.tolist(),
                'actual': self.y_test.tolist(),
                'model': voting_clf_hard
            }
        }
        
        self.models['ensemble'] = ensemble_results
        self.results['ensemble'] = ensemble_results
        
        print(f"Soft Voting Accuracy: {soft_accuracy:.4f}")
        print(f"Hard Voting Accuracy: {hard_accuracy:.4f}")
        
        return ensemble_results
    
    def create_clustering_models(self):
        """Create clustering models for pattern discovery"""
        print("\n" + "="*60)
        print("CLUSTERING MODEL TRAINING")
        print("="*60)
        
        # Use scaled features for clustering
        X_cluster = self.X_train_scaled
        
        # K-Means clustering
        kmeans_results = {}
        for n_clusters in [2, 3, 4, 5]:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X_cluster)
            
            kmeans_results[f'KMeans_{n_clusters}'] = {
                'model': kmeans,
                'cluster_labels': cluster_labels.tolist(),
                'inertia': kmeans.inertia_
            }
            print(f"KMeans_{n_clusters} - Inertia: {kmeans.inertia_:.4f}")
        
        # Hierarchical clustering
        hierarchical_results = {}
        for n_clusters in [2, 3, 4, 5]:
            agg_clustering = AgglomerativeClustering(n_clusters=n_clusters)
            cluster_labels = agg_clustering.fit_predict(X_cluster)
            
            hierarchical_results[f'Hierarchical_{n_clusters}'] = {
                'model': agg_clustering,
                'cluster_labels': cluster_labels.tolist()
            }
            print(f"Hierarchical_{n_clusters} - Completed")
        
        # PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_cluster)
        
        pca_results = {
            'model': pca,
            'transformed_features': X_pca.tolist(),
            'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
            'components': pca.components_.tolist()
        }
        
        # t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        X_tsne = tsne.fit_transform(X_cluster[:1000])  # Use subset for efficiency
        
        tsne_results = {
            'model': tsne,
            'transformed_features': X_tsne.tolist()
        }
        
        clustering_results = {
            'kmeans': kmeans_results,
            'hierarchical': hierarchical_results,
            'pca': pca_results,
            'tsne': tsne_results
        }
        
        self.models['clustering'] = clustering_results
        self.results['clustering'] = clustering_results
        
        # Save scalers and models
        joblib.dump(self.scaler, os.path.join(self.model_dir, 'clustering_scaler.pkl'))
        joblib.dump(pca, os.path.join(self.model_dir, 'pca_model.pkl'))
        
        return clustering_results
    
    def hyperparameter_tuning(self):
        """Perform hyperparameter tuning for top models"""
        print("\n" + "="*60)
        print("HYPERPARAMETER TUNING")
        print("="*60)
        
        # Define parameter grids for top models
        param_grids = {
            'RandomForest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10]
            },
            'XGBoost': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2]
            },
            'SVM': {
                'C': [0.1, 1, 10],
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto']
            }
        }
        
        tuned_results = {}
        
        # Tune Random Forest
        print("Tuning Random Forest...")
        rf_grid = GridSearchCV(
            RandomForestClassifier(random_state=42),
            param_grids['RandomForest'],
            cv=3,
            scoring='accuracy',
            n_jobs=-1
        )
        rf_grid.fit(self.X_train, self.y_train)
        rf_tuned_pred = rf_grid.predict(self.X_test)
        
        tuned_results['RandomForest_Tuned'] = {
            'model': rf_grid.best_estimator_,
            'best_params': rf_grid.best_params_,
            'best_score': rf_grid.best_score_,
            'accuracy': accuracy_score(self.y_test, rf_tuned_pred),
            'predictions': rf_tuned_pred.tolist()
        }
        
        print(f"RF Tuned - Best params: {rf_grid.best_params_}, Score: {rf_grid.best_score_:.4f}, Test Acc: {accuracy_score(self.y_test, rf_tuned_pred):.4f}")
        
        # Tune XGBoost
        print("Tuning XGBoost...")
        xgb_grid = GridSearchCV(
            xgb.XGBClassifier(random_state=42),
            param_grids['XGBoost'],
            cv=3,
            scoring='accuracy',
            n_jobs=-1
        )
        xgb_grid.fit(self.X_train, self.y_train)
        xgb_tuned_pred = xgb_grid.predict(self.X_test)
        
        tuned_results['XGBoost_Tuned'] = {
            'model': xgb_grid.best_estimator_,
            'best_params': xgb_grid.best_params_,
            'best_score': xgb_grid.best_score_,
            'accuracy': accuracy_score(self.y_test, xgb_tuned_pred),
            'predictions': xgb_tuned_pred.tolist()
        }
        
        print(f"XGB Tuned - Best params: {xgb_grid.best_params_}, Score: {xgb_grid.best_score_:.4f}, Test Acc: {accuracy_score(self.y_test, xgb_tuned_pred):.4f}")
        
        self.models['tuned'] = tuned_results
        self.results['tuned'] = tuned_results
        
        return tuned_results
    
    def generate_reports(self):
        """Generate comprehensive reports"""
        print("\n" + "="*60)
        print("GENERATING REPORTS")
        print("="*60)
        
        report_content = f"""
ANIMAL DISEASE PREDICTION ANALYSIS REPORT
========================================

Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Dataset Shape: {self.df.shape if self.df is not None else 'Unknown'}

CLASSIFICATION RESULTS:
=======================
"""
        
        # Add classification results
        if 'classification' in self.results:
            for model_name, result in self.results['classification'].items():
                if isinstance(result, dict) and 'accuracy' in result:
                    report_content += f"{model_name}:\n"
                    report_content += f"  Accuracy: {result['accuracy']:.4f}\n"
                    report_content += f"  F1 Score: {result['f1_score']:.4f}\n"
                    report_content += f"  Cross-validation: {result['cv_mean']:.4f} ± {result['cv_std']:.4f}\n\n"
        
        # Add ensemble results
        if 'ensemble' in self.results:
            report_content += "\nENSEMBLE RESULTS:\n"
            for model_name, result in self.results['ensemble'].items():
                report_content += f"{model_name}: {result['accuracy']:.4f}\n"
        
        # Add tuned results
        if 'tuned' in self.results:
            report_content += "\nHYPERPARAMETER TUNING RESULTS:\n"
            for model_name, result in self.results['tuned'].items():
                report_content += f"{model_name}: {result['accuracy']:.4f} (CV: {result['best_score']:.4f})\n"
                report_content += f"  Best Params: {result['best_params']}\n\n"
        
        # Save report
        report_path = os.path.join(self.report_dir, f'animal_disease_report_{self.timestamp}.txt')
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        print(f"Report saved to: {report_path}")
        
        # Save results as JSON
        results_json_path = os.path.join(self.report_dir, f'animal_disease_results_{self.timestamp}.json')
        with open(results_json_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        return report_path
    
    def generate_visualizations(self):
        """Generate visualizations for the results"""
        print("\n" + "="*60)
        print("GENERATING VISUALIZATIONS")
        print("="*60)
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Model Comparison Bar Chart
        if 'classification' in self.results:
            model_names = list(self.results['classification'].keys())
            accuracies = [self.results['classification'][name]['accuracy'] for name in model_names]
            
            axes[0, 0].barh(range(len(model_names)), accuracies)
            axes[0, 0].set_yticks(range(len(model_names)))
            axes[0, 0].set_yticklabels(model_names, fontsize=8)
            axes[0, 0].set_xlabel('Accuracy')
            axes[0, 0].set_title('Model Accuracy Comparison')
            
            # Add value labels
            for i, acc in enumerate(accuracies):
                axes[0, 0].text(acc + 0.01, i, f'{acc:.3f}', va='center', fontsize=7)
        
        # 2. Disease Distribution
        disease_counts = self.df['Disease'].value_counts()
        axes[0, 1].pie(disease_counts.values, labels=disease_counts.index, autopct='%1.1f%%')
        axes[0, 1].set_title('Disease Distribution')
        
        # 3. Feature Correlation Heatmap
        feature_cols = ['Age', 'Temperature']
        symptom_cols = [col for col in self.X.columns if 'encoded' in col and col != 'Disease_encoded']
        corr_cols = feature_cols + symptom_cols[:2]  # Limit for readability
        corr_data = self.X[corr_cols].corr()
        
        sns.heatmap(corr_data, annot=True, cmap='coolwarm', center=0, ax=axes[1, 0])
        axes[1, 0].set_title('Feature Correlation Heatmap')
        
        # 4. Temperature vs Age scatter by disease
        # Decode the disease labels to plot
        disease_decoded = [self.label_encoders['Disease'].inverse_transform([y])[0] for y in self.y_test[:1000]]  # Limit for performance
        scatter = axes[1, 1].scatter(
            self.X_test['Age'][:1000], 
            self.X_test['Temperature'][:1000], 
            c=pd.Categorical(disease_decoded).codes, 
            alpha=0.6, 
            cmap='viridis'
        )
        axes[1, 1].set_xlabel('Age')
        axes[1, 1].set_ylabel('Temperature')
        axes[1, 1].set_title('Age vs Temperature by Disease')
        plt.colorbar(scatter, ax=axes[1, 1])
        
        plt.tight_layout()
        viz_path = os.path.join(self.vis_dir, f'animal_disease_visualizations_{self.timestamp}.png')
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualizations saved to: {viz_path}")
        
        # Create additional visualizations for top models
        if 'classification' in self.results:
            # Confusion matrix for top model
            top_model_name = max(self.results['classification'].keys(), 
                                key=lambda x: self.results['classification'][x]['accuracy'])
            
            cm = np.array(self.results['classification'][top_model_name]['confusion_matrix'])
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=self.label_encoders['Disease'].classes_,
                       yticklabels=self.label_encoders['Disease'].classes_)
            plt.title(f'Confusion Matrix - {top_model_name}')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            cm_path = os.path.join(self.vis_dir, f'confusion_matrix_{top_model_name}_{self.timestamp}.png')
            plt.savefig(cm_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Top model confusion matrix saved to: {cm_path}")
    
    def save_models(self):
        """Save all trained models"""
        print("\n" + "="*60)
        print("SAVING MODELS")
        print("="*60)
        
        # Save all models
        for category, models in self.models.items():
            for model_name, model_data in models.items():
                if 'model' in model_data:
                    model_path = os.path.join(self.model_dir, f'{model_name.replace(" ", "_").replace("-", "_")}.pkl')
                    joblib.dump(model_data['model'], model_path)
                    print(f"Saved {model_name} to {model_path}")
        
        # Save label encoders
        for col, encoder in self.label_encoders.items():
            encoder_path = os.path.join(self.model_dir, f'{col}_label_encoder.pkl')
            joblib.dump(encoder, encoder_path)
        
        # Save preprocessing objects
        joblib.dump(self.scaler, os.path.join(self.model_dir, 'feature_scaler.pkl'))
        
        print(f"All models saved to: {self.model_dir}")
    
    def run_complete_analysis(self):
        """Run the complete analysis pipeline"""
        print("Starting comprehensive animal disease analysis...")
        
        # Load data
        self.load_data()
        
        # Preprocess data
        self.preprocess_data()
        
        # Train classification models
        classification_results = self.train_classification_models()
        
        # Create ensemble models
        ensemble_results = self.create_ensemble_models()
        
        # Create clustering models
        clustering_results = self.create_clustering_models()
        
        # Hyperparameter tuning
        tuning_results = self.hyperparameter_tuning()
        
        # Generate reports
        self.generate_reports()
        
        # Generate visualizations
        self.generate_visualizations()
        
        # Save models
        self.save_models()
        
        # Print final summary
        print("\n" + "="*60)
        print("ANALYSIS COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Models saved to: {self.model_dir}")
        print(f"Reports saved to: {self.report_dir}")
        print(f"Visualizations saved to: {self.vis_dir}")
        
        # Find best performing model
        if 'classification' in self.results:
            best_model = max(self.results['classification'].keys(), 
                           key=lambda x: self.results['classification'][x]['accuracy'])
            best_acc = self.results['classification'][best_model]['accuracy']
            print(f"\nBest Classification Model: {best_model}")
            print(f"Best Accuracy: {best_acc:.4f}")
        
        return self.results

# Main execution
if __name__ == "__main__":
    predictor = AnimalDiseasePredictor()
    results = predictor.run_complete_analysis()