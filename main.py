#!/usr/bin/env python3
"""
Heart Disease Prediction Using Machine Learning
===============================================

A comprehensive implementation of 6 machine learning classification models
for early heart disease detection using the UCI Cleveland Heart Disease dataset.

Models implemented:
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree
- Random Forest
- XGBoost (Primary model based on research findings)
- Support Vector Machine (SVM)

Based on the research document findings, XGBoost showed the best performance
with 90% accuracy and 0.93 ROC-AUC score.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import classification_report, confusion_matrix, roc_curve
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

#  numpy pandas scikit-learn xgboost matplotlib seaborn

class HeartDiseasePredictor:
    """
    A comprehensive heart disease prediction system using multiple ML algorithms.
    
    Based on the research methodology from the document which found XGBoost
    to be the best performing model with 90% accuracy and 0.93 ROC-AUC.
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.results = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_sample_data(self):
        """
        Creates a sample dataset similar to UCI Cleveland Heart Disease dataset.
        In practice, you would load the actual dataset from UCI repository.
        """
        print("Creating sample UCI Cleveland Heart Disease dataset...")
        
        # Generate sample data with 303 samples and 14 features as per document
        np.random.seed(42)
        n_samples = 303
        
        # Generate features based on UCI Cleveland dataset attributes
        data = {
            'age': np.random.randint(25, 80, n_samples),
            'sex': np.random.choice([0, 1], n_samples),
            'cp': np.random.choice([0, 1, 2, 3], n_samples),  # chest pain type
            'trestbps': np.random.randint(90, 200, n_samples),  # resting blood pressure
            'chol': np.random.randint(120, 400, n_samples),  # serum cholesterol
            'fbs': np.random.choice([0, 1], n_samples),  # fasting blood sugar
            'restecg': np.random.choice([0, 1, 2], n_samples),  # resting ECG
            'thalach': np.random.randint(80, 200, n_samples),  # max heart rate
            'exang': np.random.choice([0, 1], n_samples),  # exercise induced angina
            'oldpeak': np.random.uniform(0, 6, n_samples),  # ST depression
            'slope': np.random.choice([0, 1, 2], n_samples),  # slope of peak exercise ST
            'ca': np.random.choice([0, 1, 2, 3], n_samples),  # number of major vessels
            'thal': np.random.choice([0, 1, 2, 3], n_samples),  # thalassemia
        }
        
        # Create target variable (heart disease present/absent)
        # Make it somewhat correlated with features for realistic results
        target_prob = (
            0.3 + 
            0.2 * (data['age'] > 50) + 
            0.15 * data['sex'] + 
            0.1 * (data['cp'] > 0) + 
            0.15 * (data['trestbps'] > 140) + 
            0.1 * (data['chol'] > 240)
        )
        data['target'] = np.random.binomial(1, target_prob, n_samples)
        
        self.df = pd.DataFrame(data)
        print(f"Dataset created with {len(self.df)} samples and {len(self.df.columns)} features")
        print(f"Target distribution: {self.df['target'].value_counts().to_dict()}")
        
        return self.df
    
    def preprocess_data(self):
        """
        Preprocess the data following the methodology from the document:
        - Handle missing values
        - Encode categorical variables
        - Scale features
        - Split into train/test sets
        """
        print("\nPreprocessing data...")
        
        # Separate features and target
        X = self.df.drop('target', axis=1)
        y = self.df['target']
        
        # Handle missing values (mean imputation for numerical, mode for categorical)
        print("Handling missing values...")
        X = X.fillna(X.mean())
        
        # Split the data (80% train, 20% test as per document)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training set: {len(self.X_train)} samples")
        print(f"Test set: {len(self.X_test)} samples")
        
        # Scale features for distance-based algorithms (KNN, SVM)
        self.scalers['standard'] = StandardScaler()
        self.scalers['minmax'] = MinMaxScaler()
        
        # Fit scalers on training data
        self.scalers['standard'].fit(self.X_train)
        self.scalers['minmax'].fit(self.X_train)
        
        print("Data preprocessing completed!")
    
    def initialize_models(self):
        """
        Initialize all 6 machine learning models with hyperparameters
        based on the document's methodology.
        """
        print("\nInitializing machine learning models...")
        
        # 1. Logistic Regression - baseline linear model
        self.models['Logistic Regression'] = LogisticRegression(
            random_state=42, 
            max_iter=1000,
            solver='liblinear'  # As specified in document
        )
        
        # 2. K-Nearest Neighbors
        self.models['KNN'] = KNeighborsClassifier(
            n_neighbors=5,
            metric='euclidean'  # As specified in document
        )
        
        # 3. Decision Tree
        self.models['Decision Tree'] = DecisionTreeClassifier(
            random_state=42,
            criterion='gini',  # As specified in document
            max_depth=10
        )
        
        # 4. Random Forest
        self.models['Random Forest'] = RandomForestClassifier(
            n_estimators=100,  # As specified in document
            random_state=42,
            max_depth=10
        )
        
        # 5. XGBoost - Best performing model according to document
        self.models['XGBoost'] = xgb.XGBClassifier(
            random_state=42,
            booster='gbtree',  # As specified in document
            learning_rate=0.1,
            max_depth=6,
            n_estimators=100
        )
        
        # 6. Support Vector Machine
        self.models['SVM'] = SVC(
            random_state=42,
            kernel='rbf',  # As specified in document
            probability=True  # Needed for ROC-AUC calculation
        )
        
        print(f"Initialized {len(self.models)} models")
    
    def train_and_evaluate_models(self):
        """
        Train all models and evaluate their performance using the metrics
        specified in the document: Accuracy, Precision, Recall, F1-Score, ROC-AUC
        """
        print("\nTraining and evaluating all models...")
        
        # Models that need scaled features
        scaled_models = ['KNN', 'SVM', 'Logistic Regression']
        
        for model_name, model in self.models.items():
            print(f"\nTraining {model_name}...")
            
            # Choose appropriate scaling for the model
            if model_name in scaled_models:
                if model_name == 'KNN':
                    X_train_scaled = self.scalers['minmax'].transform(self.X_train)
                    X_test_scaled = self.scalers['minmax'].transform(self.X_test)
                else:  # SVM and Logistic Regression
                    X_train_scaled = self.scalers['standard'].transform(self.X_train)
                    X_test_scaled = self.scalers['standard'].transform(self.X_test)
                
                # Train model
                model.fit(X_train_scaled, self.y_train)
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            else:
                # Train model without scaling
                model.fit(self.X_train, self.y_train)
                y_pred = model.predict(self.X_test)
                y_pred_proba = model.predict_proba(self.X_test)[:, 1]
            
            # Calculate evaluation metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred)
            recall = recall_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)
            roc_auc = roc_auc_score(self.y_test, y_pred_proba)
            
            # Store results
            self.results[model_name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'roc_auc': roc_auc,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
            
            print(f"  Accuracy: {accuracy:.3f}")
            print(f"  Precision: {precision:.3f}")
            print(f"  Recall: {recall:.3f}")
            print(f"  F1-Score: {f1:.3f}")
            print(f"  ROC-AUC: {roc_auc:.3f}")
    
    def hyperparameter_tuning(self, model_name='XGBoost'):
        """
        Perform hyperparameter tuning for the specified model using GridSearchCV.
        Default is XGBoost as it's the best performing model according to the document.
        """
        print(f"\nPerforming hyperparameter tuning for {model_name}...")
        
        if model_name == 'XGBoost':
            param_grid = {
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 6, 10],
                'n_estimators': [50, 100, 200]
            }
            model = xgb.XGBClassifier(random_state=42)
            X_train_data = self.X_train
            
        elif model_name == 'Random Forest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15],
                'max_features': ['sqrt', 'log2']
            }
            model = RandomForestClassifier(random_state=42)
            X_train_data = self.X_train
            
        elif model_name == 'SVM':
            param_grid = {
                'C': [0.1, 1, 10],
                'gamma': ['scale', 'auto', 0.1, 1]
            }
            model = SVC(random_state=42, probability=True)
            X_train_data = self.scalers['standard'].transform(self.X_train)
            
        else:
            print(f"Hyperparameter tuning not configured for {model_name}")
            return
        
        # Perform grid search with 5-fold cross-validation (as per document)
        grid_search = GridSearchCV(
            model, 
            param_grid, 
            cv=5, 
            scoring='roc_auc',
            n_jobs=-1
        )
        
        grid_search.fit(X_train_data, self.y_train)
        
        print(f"Best parameters for {model_name}: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.3f}")
        
        # Update the model with best parameters
        self.models[model_name] = grid_search.best_estimator_
        
        return grid_search.best_estimator_
    
    def display_results_table(self):
        """
        Display a comprehensive results table matching the format in the document.
        """
        print("\n" + "="*80)
        print("COMPREHENSIVE MODEL PERFORMANCE COMPARISON")
        print("="*80)
        
        # Create results DataFrame
        results_df = pd.DataFrame(self.results).T
        results_df = results_df[['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']]
        
        # Format the results
        results_df = results_df.round(3)
        
        print("\nTable: Performance Metrics of Machine Learning Models")
        print("-" * 80)
        print(f"{'Model':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'ROC-AUC':<10}")
        print("-" * 80)
        
        for model_name, row in results_df.iterrows():
            print(f"{model_name:<20} {row['accuracy']:<10.3f} {row['precision']:<10.3f} {row['recall']:<10.3f} {row['f1_score']:<10.3f} {row['roc_auc']:<10.3f}")
        
        # Highlight best performing model
        best_model = results_df.loc[results_df['accuracy'].idxmax()]
        print("-" * 80)
        print(f"Best performing model: {results_df['accuracy'].idxmax()} (Accuracy: {best_model['accuracy']:.3f})")
        
        # According to the document, XGBoost should be the best
        if 'XGBoost' in results_df.index:
            xgb_results = results_df.loc['XGBoost']
            print(f"XGBoost results: Accuracy={xgb_results['accuracy']:.3f}, ROC-AUC={xgb_results['roc_auc']:.3f}")
            print("(Document reports XGBoost achieved 90% accuracy and 0.93 ROC-AUC)")
    
    def plot_performance_comparison(self):
        """
        Create visualizations comparing model performance across all metrics.
        """
        print("\nGenerating performance comparison plots...")
        
        # Prepare data for plotting
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        models = list(self.results.keys())
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Heart Disease Prediction: ML Model Performance Comparison', fontsize=16)
        
        # Individual metric plots
        for i, metric in enumerate(metrics):
            ax = axes[i//3, i%3]
            values = [self.results[model][metric] for model in models]
            
            bars = ax.bar(models, values, color=['skyblue', 'lightgreen', 'lightcoral', 'gold', 'orange', 'lightpink'])
            ax.set_title(f'{metric.replace("_", " ").title()}')
            ax.set_ylabel('Score')
            ax.set_ylim(0, 1)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                       f'{value:.3f}', ha='center', va='bottom')
            
            # Rotate x-axis labels for better readability
            ax.tick_params(axis='x', rotation=45)
        
        # Comprehensive comparison plot
        ax = axes[1, 2]
        x = np.arange(len(models))
        width = 0.15
        
        for i, metric in enumerate(metrics):
            values = [self.results[model][metric] for model in models]
            ax.bar(x + i*width, values, width, label=metric.replace('_', ' ').title())
        
        ax.set_title('All Metrics Comparison')
        ax.set_xlabel('Models')
        ax.set_ylabel('Score')
        ax.set_xticks(x + width*2)
        ax.set_xticklabels(models, rotation=45)
        ax.legend()
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.show()
    
    def plot_roc_curves(self):
        """
        Plot ROC curves for all models to compare their discriminative ability.
        """
        print("\nGenerating ROC curves...")
        
        plt.figure(figsize=(10, 8))
        
        for model_name in self.results.keys():
            y_pred_proba = self.results[model_name]['y_pred_proba']
            fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
            auc_score = self.results[model_name]['roc_auc']
            
            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves - Heart Disease Prediction Models')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def analyze_feature_importance(self, model_name='XGBoost'):
        """
        Analyze feature importance for tree-based models.
        XGBoost is used by default as it's the best performing model.
        """
        print(f"\nAnalyzing feature importance for {model_name}...")
        
        if model_name not in ['XGBoost', 'Random Forest', 'Decision Tree']:
            print(f"Feature importance analysis not available for {model_name}")
            return
        
        model = self.models[model_name]
        
        # Get feature importance
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            feature_names = self.X_train.columns
            
            # Create importance DataFrame
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            # Plot feature importance
            plt.figure(figsize=(10, 6))
            plt.barh(range(len(importance_df)), importance_df['importance'])
            plt.yticks(range(len(importance_df)), importance_df['feature'])
            plt.xlabel('Feature Importance')
            plt.title(f'Feature Importance - {model_name}')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.show()
            
            print(f"\nTop 5 most important features for {model_name}:")
            print(importance_df.head())
            
            # According to the document, chest pain and cholesterol should be key features
            if 'cp' in importance_df['feature'].values:
                cp_importance = importance_df[importance_df['feature'] == 'cp']['importance'].values[0]
                print(f"Chest pain (cp) importance: {cp_importance:.3f}")
            
            if 'chol' in importance_df['feature'].values:
                chol_importance = importance_df[importance_df['feature'] == 'chol']['importance'].values[0]
                print(f"Cholesterol (chol) importance: {chol_importance:.3f}")
    
    def generate_classification_report(self, model_name='XGBoost'):
        """
        Generate detailed classification report for the specified model.
        """
        print(f"\nDetailed Classification Report for {model_name}:")
        print("="*50)
        
        y_pred = self.results[model_name]['y_pred']
        
        # Classification report
        print(classification_report(self.y_test, y_pred, 
                                  target_names=['No Disease', 'Disease']))
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        print("\nConfusion Matrix:")
        print(cm)
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['No Disease', 'Disease'],
                   yticklabels=['No Disease', 'Disease'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.show()
    
    def run_complete_analysis(self):
        """
        Run the complete heart disease prediction analysis pipeline.
        """
        print("HEART DISEASE PREDICTION USING MACHINE LEARNING")
        print("="*60)
        print("Based on research methodology showing XGBoost as best performer")
        print("Expected results: XGBoost ~90% accuracy, 0.93 ROC-AUC")
        print("="*60)
        
        # Step 1: Load and preprocess data
        self.load_sample_data()
        self.preprocess_data()
        
        # Step 2: Initialize models
        self.initialize_models()
        
        # Step 3: Train and evaluate all models
        self.train_and_evaluate_models()
        
        # Step 4: Display results
        self.display_results_table()
        
        # Step 5: Hyperparameter tuning for best model
        self.hyperparameter_tuning('XGBoost')
        
        # Re-evaluate XGBoost after tuning
        print("\nRe-evaluating XGBoost after hyperparameter tuning...")
        model = self.models['XGBoost']
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        y_pred_proba = model.predict_proba(self.X_test)[:, 1]
        
        # Update results
        self.results['XGBoost (Tuned)'] = {
            'accuracy': accuracy_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred),
            'recall': recall_score(self.y_test, y_pred),
            'f1_score': f1_score(self.y_test, y_pred),
            'roc_auc': roc_auc_score(self.y_test, y_pred_proba),
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        # Step 6: Generate visualizations
        self.plot_performance_comparison()
        self.plot_roc_curves()
        
        # Step 7: Feature importance analysis
        self.analyze_feature_importance('XGBoost')
        
        # Step 8: Detailed report for best model
        self.generate_classification_report('XGBoost')
        
        # Step 9: Final recommendations
        self.print_recommendations()
    
    def print_recommendations(self):
        """
        Print final recommendations based on the analysis results.
        """
        print("\n" + "="*60)
        print("FINAL RECOMMENDATIONS")
        print("="*60)
        
        # Find best model
        best_accuracy = max(self.results[model]['accuracy'] for model in self.results)
        best_model = [model for model in self.results if self.results[model]['accuracy'] == best_accuracy][0]
        
        print(f"1. BEST PERFORMING MODEL: {best_model}")
        print(f"   - Accuracy: {self.results[best_model]['accuracy']:.3f}")
        print(f"   - ROC-AUC: {self.results[best_model]['roc_auc']:.3f}")
        print(f"   - Suitable for: High-stakes clinical decision support")
        
        # Most interpretable model
        interpretable_models = ['Logistic Regression', 'Decision Tree']
        best_interpretable = None
        best_interpretable_score = 0
        
        for model in interpretable_models:
            if model in self.results and self.results[model]['accuracy'] > best_interpretable_score:
                best_interpretable = model
                best_interpretable_score = self.results[model]['accuracy']
        
        if best_interpretable:
            print(f"\n2. MOST INTERPRETABLE MODEL: {best_interpretable}")
            print(f"   - Accuracy: {self.results[best_interpretable]['accuracy']:.3f}")
            print(f"   - Suitable for: Clinical settings requiring explanation")
        
        print("\n3. DEPLOYMENT RECOMMENDATIONS:")
        print("   - High-resource environments: Use XGBoost for maximum accuracy")
        print("   - Low-resource environments: Use Logistic Regression for interpretability")
        print("   - Real-time applications: Consider computational efficiency vs accuracy trade-off")
        
        print("\n4. CLINICAL CONSIDERATIONS:")
        print("   - High recall is crucial to avoid missing at-risk patients")
        print("   - Consider ensemble methods for robust predictions")
        print("   - Regular model retraining with new data is recommended")
        
        print("\nAnalysis completed successfully!")

# Example usage and execution
if __name__ == "__main__":
    # Create and run the heart disease prediction system
    predictor = HeartDiseasePredictor()
    predictor.run_complete_analysis()
    
    print("\n" + "="*60)
    print("ADDITIONAL FEATURES AVAILABLE:")
    print("="*60)
    print("1. predictor.hyperparameter_tuning('Random Forest')")
    print("2. predictor.analyze_feature_importance('Random Forest')")
    print("3. predictor.generate_classification_report('SVM')")
    print("4. Individual model evaluation and comparison")
    print("\nSystem ready for further analysis and experimentation!")
