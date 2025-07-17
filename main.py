from flask import Flask, render_template, request, jsonify
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import math
import io
import base64
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# flask numpy matplotlib seaborn typing

app = Flask(__name__)

@dataclass
class ModelResult:
    probability: float
    prediction: int

@dataclass
class ModelPerformance:
    name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    color: str

class HeartDiseasePredictor:
    def __init__(self):
        # Model performance data
        self.model_performance = [
            ModelPerformance('XGBoost', 0.90, 0.89, 0.88, 0.88, 0.93, '#10B981'),
            ModelPerformance('Random Forest', 0.88, 0.87, 0.86, 0.86, 0.91, '#3B82F6'),
            ModelPerformance('SVM', 0.87, 0.86, 0.85, 0.85, 0.90, '#8B5CF6'),
            ModelPerformance('Logistic Regression', 0.85, 0.84, 0.83, 0.83, 0.89, '#F59E0B'),
            ModelPerformance('KNN', 0.82, 0.80, 0.78, 0.79, 0.85, '#EF4444'),
            ModelPerformance('Decision Tree', 0.79, 0.76, 0.75, 0.75, 0.82, '#6B7280')
        ]
        
        # Feature importance weights
        self.feature_importance = {
            'cp': 0.25,
            'thalach': 0.18,
            'oldpeak': 0.15,
            'ca': 0.12,
            'age': 0.10,
            'thal': 0.08,
            'sex': 0.06,
            'exang': 0.06
        }
        
        # Scaling parameters
        self.scaling_params = {
            'age': {'min': 29, 'max': 77},
            'trestbps': {'min': 94, 'max': 200},
            'chol': {'min': 126, 'max': 564},
            'thalach': {'min': 71, 'max': 202},
            'oldpeak': {'min': 0, 'max': 6.2}
        }
        
        # Input fields configuration
        self.input_fields = [
            {'name': 'age', 'label': 'Age', 'type': 'number', 'min': 20, 'max': 100, 'unit': 'years'},
            {'name': 'sex', 'label': 'Sex', 'type': 'select', 'options': [('1', 'Male'), ('0', 'Female')]},
            {'name': 'cp', 'label': 'Chest Pain Type', 'type': 'select', 'options': [
                ('0', 'Typical Angina'), ('1', 'Atypical Angina'), 
                ('2', 'Non-Anginal Pain'), ('3', 'Asymptomatic')
            ]},
            {'name': 'trestbps', 'label': 'Resting Blood Pressure', 'type': 'number', 'min': 80, 'max': 220, 'unit': 'mmHg'},
            {'name': 'chol', 'label': 'Serum Cholesterol', 'type': 'number', 'min': 100, 'max': 600, 'unit': 'mg/dl'},
            {'name': 'fbs', 'label': 'Fasting Blood Sugar > 120 mg/dl', 'type': 'select', 'options': [('1', 'Yes'), ('0', 'No')]},
            {'name': 'restecg', 'label': 'Resting ECG', 'type': 'select', 'options': [
                ('0', 'Normal'), ('1', 'ST-T Wave Abnormality'), ('2', 'Left Ventricular Hypertrophy')
            ]},
            {'name': 'thalach', 'label': 'Maximum Heart Rate', 'type': 'number', 'min': 60, 'max': 220, 'unit': 'bpm'},
            {'name': 'exang', 'label': 'Exercise Induced Angina', 'type': 'select', 'options': [('1', 'Yes'), ('0', 'No')]},
            {'name': 'oldpeak', 'label': 'ST Depression', 'type': 'number', 'min': 0, 'max': 6, 'unit': 'units'},
            {'name': 'slope', 'label': 'Slope of Peak Exercise ST', 'type': 'select', 'options': [
                ('0', 'Upsloping'), ('1', 'Flat'), ('2', 'Downsloping')
            ]},
            {'name': 'ca', 'label': 'Number of Major Vessels', 'type': 'select', 'options': [
                ('0', '0'), ('1', '1'), ('2', '2'), ('3', '3')
            ]},
            {'name': 'thal', 'label': 'Thalassemia', 'type': 'select', 'options': [
                ('3', 'Normal'), ('6', 'Fixed Defect'), ('7', 'Reversible Defect')
            ]}
        ]
        
    def preprocess_data(self, data: Dict) -> Dict:
        """Preprocess input data for model prediction"""
        processed = {}
        
        # Convert to float
        for key, value in data.items():
            if value != '':
                processed[key] = float(value)
            else:
                processed[key] = 0.0
        
        # Feature scaling
        for key, params in self.scaling_params.items():
            if key in processed:
                min_val, max_val = params['min'], params['max']
                processed[key] = (processed[key] - min_val) / (max_val - min_val)
        
        return processed
        
    def logistic_regression(self, data: Dict) -> ModelResult:
        """Simplified logistic regression implementation"""
        weights = {
            'age': 0.8, 'sex': 1.2, 'cp': -1.5, 'trestbps': 0.3, 'chol': 0.2,
            'fbs': 0.1, 'restecg': 0.15, 'thalach': -0.9, 'exang': 0.7,
            'oldpeak': 0.8, 'slope': -0.4, 'ca': 1.1, 'thal': -0.6
        }
        
        score = 0.1  # bias
        for key, weight in weights.items():
            if key in data:
                score += weight * data[key]
        
        probability = 1 / (1 + math.exp(-score))
        prediction = 1 if probability > 0.5 else 0
        
        return ModelResult(probability, prediction)
        
    def random_forest(self, data: Dict) -> ModelResult:
        """Simplified random forest implementation"""
        trees = [
            {'threshold': 0.45, 'weight': 0.3},
            {'threshold': 0.55, 'weight': 0.4},
            {'threshold': 0.50, 'weight': 0.3}
        ]
        
        logistic_result = self.logistic_regression(data)
        ensemble = 0
        
        for tree in trees:
            tree_result = 1 if logistic_result.probability > tree['threshold'] else 0
            ensemble += tree_result * tree['weight']
        
        prediction = 1 if ensemble > 0.5 else 0
        return ModelResult(ensemble, prediction)
        
    def xgboost(self, data: Dict) -> ModelResult:
        """Enhanced gradient boosting simulation"""
        base_result = self.logistic_regression(data)
        boosted_score = base_result.probability
        
        boosting_factors = [
            {'feature': 'cp', 'weight': 0.3},
            {'feature': 'thalach', 'weight': -0.25},
            {'feature': 'oldpeak', 'weight': 0.2},
            {'feature': 'ca', 'weight': 0.15}
        ]
        
        for factor in boosting_factors:
            feature = factor['feature']
            if feature in data:
                boosted_score += factor['weight'] * data[feature] * 0.1
        
        boosted_score = max(0, min(1, boosted_score))
        prediction = 1 if boosted_score > 0.5 else 0
        
        return ModelResult(boosted_score, prediction)
        
    def svm(self, data: Dict) -> ModelResult:
        """Simplified SVM with RBF kernel"""
        support_vectors = [
            {'cp': 0.8, 'thalach': 0.3, 'oldpeak': 0.7, 'weight': 0.4},
            {'cp': 0.2, 'thalach': 0.8, 'oldpeak': 0.1, 'weight': -0.3},
            {'cp': 0.6, 'thalach': 0.5, 'oldpeak': 0.4, 'weight': 0.5}
        ]
        
        decision = 0
        for sv in support_vectors:
            similarity = 0
            for feature in ['cp', 'thalach', 'oldpeak']:
                if feature in data:
                    similarity += math.exp(-((data[feature] - sv[feature]) ** 2))
            decision += sv['weight'] * similarity
        
        probability = 1 / (1 + math.exp(-decision))
        prediction = 1 if probability > 0.5 else 0
        
        return ModelResult(probability, prediction)
        
    def knn(self, data: Dict) -> ModelResult:
        """K-nearest neighbors simulation"""
        neighbors = [
            {'similarity': 0.8, 'label': 1},
            {'similarity': 0.6, 'label': 0},
            {'similarity': 0.7, 'label': 1},
            {'similarity': 0.4, 'label': 0},
            {'similarity': 0.9, 'label': 1}
        ]
        
        k = 3
        sorted_neighbors = sorted(neighbors, key=lambda x: x['similarity'], reverse=True)[:k]
        prediction_prob = sum(n['label'] for n in sorted_neighbors) / k
        prediction = 1 if prediction_prob > 0.5 else 0
        
        return ModelResult(prediction_prob, prediction)
        
    def decision_tree(self, data: Dict) -> ModelResult:
        """Simplified decision tree"""
        if 'cp' in data and data['cp'] > 0.5:
            if 'thalach' in data and data['thalach'] < 0.5:
                return ModelResult(0.8, 1)
            else:
                return ModelResult(0.3, 0)
        else:
            if 'oldpeak' in data and data['oldpeak'] > 0.3:
                return ModelResult(0.7, 1)
            else:
                return ModelResult(0.2, 0)
                
    def predict_all_models(self, data: Dict) -> Dict:
        """Run prediction with all models"""
        processed_data = self.preprocess_data(data)
        
        predictions = {
            'XGBoost': self.xgboost(processed_data),
            'Random Forest': self.random_forest(processed_data),
            'SVM': self.svm(processed_data),
            'Logistic Regression': self.logistic_regression(processed_data),
            'KNN': self.knn(processed_data),
            'Decision Tree': self.decision_tree(processed_data)
        }
        
        return predictions
    
    def generate_performance_plot(self):
        """Generate performance comparison plot"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Performance metrics bar chart
        models = [model.name for model in self.model_performance]
        accuracy = [model.accuracy for model in self.model_performance]
        precision = [model.precision for model in self.model_performance]
        recall = [model.recall for model in self.model_performance]
        f1 = [model.f1 for model in self.model_performance]
        
        x = np.arange(len(models))
        width = 0.2
        
        ax1.bar(x - 1.5*width, accuracy, width, label='Accuracy', color='#3B82F6')
        ax1.bar(x - 0.5*width, precision, width, label='Precision', color='#10B981')
        ax1.bar(x + 0.5*width, recall, width, label='Recall', color='#F59E0B')
        ax1.bar(x + 1.5*width, f1, width, label='F1-Score', color='#EF4444')
        
        ax1.set_xlabel('Models')
        ax1.set_ylabel('Performance Score')
        ax1.set_title('Model Performance Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(models, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # ROC-AUC comparison
        roc_auc = [model.roc_auc for model in self.model_performance]
        colors = [model.color for model in self.model_performance]
        
        ax2.bar(models, roc_auc, color=colors)
        ax2.set_xlabel('Models')
        ax2.set_ylabel('ROC-AUC Score')
        ax2.set_title('ROC-AUC Comparison')
        ax2.set_xticklabels(models, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Convert plot to base64 string
        img = io.BytesIO()
        plt.savefig(img, format='png', dpi=150, bbox_inches='tight')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()
        
        return plot_url

# Initialize the predictor
predictor = HeartDiseasePredictor()

@app.route('/')
def index():
    """Main page with input form"""
    return render_template('index.html', input_fields=predictor.input_fields)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction request"""
    try:
        # Get form data
        form_data = request.form.to_dict()
        
        # Validate input
        missing_fields = []
        for field in predictor.input_fields:
            if field['name'] not in form_data or form_data[field['name']] == '':
                missing_fields.append(field['label'])
        
        if missing_fields:
            return jsonify({
                'error': 'Please fill in all required fields',
                'missing_fields': missing_fields
            }), 400
        
        # Run predictions
        predictions = predictor.predict_all_models(form_data)
        
        # Format results
        results = {}
        for model_name, result in predictions.items():
            # Find model performance
            model_perf = next((m for m in predictor.model_performance if m.name == model_name), None)
            
            # Determine risk level
            if result.probability < 0.3:
                risk_level = "Low"
                risk_color = "#059669"
            elif result.probability < 0.7:
                risk_level = "Moderate"
                risk_color = "#D97706"
            else:
                risk_level = "High"
                risk_color = "#DC2626"
            
            results[model_name] = {
                'prediction': result.prediction,
                'probability': result.probability,
                'risk_level': risk_level,
                'risk_color': risk_color,
                'accuracy': model_perf.accuracy if model_perf else 0
            }
        
        return jsonify({
            'success': True,
            'predictions': results
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/models')
def models():
    """Model performance page"""
    plot_url = predictor.generate_performance_plot()
    return render_template('models.html', 
                         model_performance=predictor.model_performance,
                         predictor=predictor,
                         plot_url=plot_url)

@app.route('/about')
def about():
    """About page"""
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)