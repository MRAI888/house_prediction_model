from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import os
import json

app = Flask(__name__, static_folder='../frontend', template_folder='../frontend')
CORS(app)

base_dir = os.path.dirname(__file__)
models_dir = os.path.join(base_dir, '..', 'models')
notebook_models_dir = os.path.join(base_dir, '..', 'models_saved')


# Load model artifacts – try notebook version first, fallback to simple

model = None
preprocessor = None
scaler = None
features = None
use_notebook_model = False

# Check for notebook saved model
notebook_model_path = os.path.join(notebook_models_dir, 'best_model.pkl')
notebook_preprocessor_path = os.path.join(notebook_models_dir, 'preprocessor.pkl')
notebook_keras_path = os.path.join(notebook_models_dir, 'best_model.keras')

# Check for simple model
simple_model_path = os.path.join(models_dir, 'xgboost_model.pkl')
simple_scaler_path = os.path.join(models_dir, 'scaler.pkl')
simple_features_path = os.path.join(models_dir, 'features.pkl')

if os.path.exists(notebook_model_path) and os.path.exists(notebook_preprocessor_path):
    print("[INFO] Loading notebook-trained model and preprocessor...")
    model = joblib.load(notebook_model_path)
    preprocessor = joblib.load(notebook_preprocessor_path)
    use_notebook_model = True
elif os.path.exists(notebook_keras_path) and os.path.exists(notebook_preprocessor_path):
    print("[INFO] Loading notebook-trained Keras model and preprocessor...")
    import tensorflow as tf
    model = tf.keras.models.load_model(notebook_keras_path)
    preprocessor = joblib.load(notebook_preprocessor_path)
    use_notebook_model = True
elif os.path.exists(simple_model_path) and os.path.exists(simple_scaler_path) and os.path.exists(simple_features_path):
    print("[INFO] Loading simple XGBoost model...")
    model = joblib.load(simple_model_path)
    scaler = joblib.load(simple_scaler_path)
    features = joblib.load(simple_features_path)
    use_notebook_model = False
else:
    raise FileNotFoundError("No trained model found. Please run train.py or the notebook first.")

print(f"[INFO] Using {'notebook' if use_notebook_model else 'simple'} model.")


# Helper: Get location encodings (for simple model only)

def get_town_encoding(town):
    # In production, load from a saved mapping file
    encodings = {'London': 550000, 'Manchester': 220000, 'Birmingham': 200000}
    return encodings.get(town, 250000)

def get_county_encoding(county):
    encodings = {'Greater London': 600000, 'Greater Manchester': 230000}
    return encodings.get(county, 280000)


# Routes

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        if use_notebook_model:
            # Notebook model expects raw features as a DataFrame
            # Build a single-row DataFrame with all expected columns
            # The preprocessor will handle missing columns (if any) via handle_unknown='ignore'
            input_dict = {}
            
            # Map frontend fields to expected column names (adjust based on your dataset)
            input_dict['property_type'] = data.get('property_type', 'D')
            input_dict['new_build'] = int(data.get('new_build', 0))
            input_dict['freehold'] = int(data.get('freehold', 1))
            input_dict['Town'] = data.get('Town', 'London')
            input_dict['County'] = data.get('County', 'Greater London')
            
            # Date features
            date_str = data.get('prediction_date', '')
            if date_str:
                d = pd.to_datetime(date_str)
                input_dict['year'] = d.year
                input_dict['month'] = d.month
            else:
                input_dict['year'] = 2024
                input_dict['month'] = 1
            
            # Market indicators (all the numeric fields from the form)
            numeric_fields = [
                'avg_1m_change', 'avg_12m_change', 'salesvolume',
                'detachedprice', 'detached1mpctchange', 'detached12mpctchange',
                'semidetachedprice', 'semidetached1mpctchange', 'semidetached12mpctchange',
                'terracedprice', 'terraced1mpctchange', 'terraced12mpctchange',
                'flatprice', 'flat1mpctchange', 'flat12mpctchange'
            ]
            for field in numeric_fields:
                input_dict[field] = float(data.get(field, 0))
            
            # Add any other features the model expects (fill with 0 or median)
            # For simplicity, we rely on the preprocessor's handle_unknown for categoricals
            input_df = pd.DataFrame([input_dict])
            
            # Preprocess
            processed = preprocessor.transform(input_df)
            pred = model.predict(processed)[0]
            if hasattr(pred, 'flatten'):
                pred = pred.flatten()[0]
        else:
            # Simple model: build feature vector matching features list
            input_dict = {col: 0 for col in features}
            
            # One-hot property type
            prop_type = data.get('property_type', 'D')
            input_dict['type_D'] = 1 if prop_type == 'D' else 0
            input_dict['type_S'] = 1 if prop_type == 'S' else 0
            input_dict['type_T'] = 1 if prop_type == 'T' else 0
            input_dict['type_F'] = 1 if prop_type == 'F' else 0
            
            input_dict['new_build'] = int(data.get('new_build', 0))
            input_dict['freehold'] = int(data.get('freehold', 1))
            
            # Date derived
            date_str = data.get('prediction_date', '')
            if date_str:
                d = pd.to_datetime(date_str)
                input_dict['year'] = d.year
                input_dict['month'] = d.month
                input_dict['quarter'] = (d.month - 1) // 3 + 1
            else:
                input_dict['year'] = 2024
                input_dict['month'] = 1
                input_dict['quarter'] = 1
            
            # Location encodings
            input_dict['Town_enc'] = get_town_encoding(data.get('Town', 'London'))
            input_dict['County_enc'] = get_county_encoding(data.get('County', 'Greater London'))
            
            # Market indicators
            numeric_fields = [
                'avg_1m_change', 'avg_12m_change', 'salesvolume',
                'detachedprice', 'detached1mpctchange', 'detached12mpctchange',
                'semidetachedprice', 'semidetached1mpctchange', 'semidetached12mpctchange',
                'terracedprice', 'terraced1mpctchange', 'terraced12mpctchange',
                'flatprice', 'flat1mpctchange', 'flat12mpctchange'
            ]
            for field in numeric_fields:
                if field in input_dict:
                    input_dict[field] = float(data.get(field, 0))
            
            input_df = pd.DataFrame([input_dict])[features]
            scaled = scaler.transform(input_df)
            pred = model.predict(scaled)[0]
        
        return jsonify({'predicted_price': float(pred)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/drift_report', methods=['GET'])
def drift_report():
    # In production, compute from recent data
    report = {
        'Price': {'PSI': 0.08, 'Alert': False},
        'new_build': {'PSI': 0.12, 'Alert': True},
        'property_type': {'PSI': 0.05, 'Alert': False}
    }
    return jsonify(report)

@app.route('/api/model_metrics', methods=['GET'])
def api_model_metrics():
    """Return model performance metrics from the saved results."""
    try:
        # Try notebook results first
        results_path = os.path.join(notebook_models_dir, 'model_results.csv')
        if not os.path.exists(results_path):
            results_path = os.path.join(base_dir, '..', 'results', 'model_comparison.csv')
        
        if os.path.exists(results_path):
            df = pd.read_csv(results_path)
            # Expect columns: Model, MAE, RMSE, R2
            return jsonify({
                'labels': df['Model'].tolist() if 'Model' in df.columns else df.iloc[:,0].tolist(),
                'r2': df['R2'].tolist() if 'R2' in df.columns else [],
                'mae': df['MAE'].tolist() if 'MAE' in df.columns else [],
                'rmse': df['RMSE'].tolist() if 'RMSE' in df.columns else []
            })
        else:
            # Fallback
            return jsonify({
                'labels': ['Linear Reg', 'Decision Tree', 'Random Forest', 'Gradient Boost', 'XGBoost'],
                'r2': [0.62, 0.71, 0.84, 0.85, 0.88],
                'mae': [48230, 34500, 25800, 24100, 22300],
                'rmse': [72100, 56200, 41900, 39500, 36200]
            })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/confusion_matrix', methods=['GET'])
def api_confusion_matrix():
    # Simulated – in production compute from test predictions
    return jsonify({
        'labels': ['Low (<£150k)', 'Mid (£150k-400k)', 'High (>£400k)'],
        'matrix': [
            [1245, 89, 12],
            [67, 1102, 45],
            [8, 34, 980]
        ]
    })

@app.route('/api/feature_importance', methods=['GET'])
def api_feature_importance():
    """Return feature importance from the model if available."""
    try:
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            # Get feature names
            if use_notebook_model and preprocessor:
                # Extract feature names from preprocessor
                feature_names = []
                for name, trans, cols in preprocessor.transformers_:
                    if name == 'num':
                        feature_names.extend(cols)
                    elif name == 'cat':
                        # OneHotEncoder creates names
                        if hasattr(trans, 'get_feature_names_out'):
                            feature_names.extend(trans.get_feature_names_out(cols))
                        else:
                            feature_names.extend([f"{col}_{val}" for col in cols for val in ['?']])
            else:
                feature_names = features if features else [f'feature_{i}' for i in range(len(importance))]
            
            # Sort and return top 8
            indices = np.argsort(importance)[::-1][:8]
            return jsonify({
                'features': [str(feature_names[i]) for i in indices],
                'values': importance[indices].tolist()
            })
        else:
            return jsonify({
                'features': ['County_enc', 'floor_area', 'type_D', 'new_build', 'bedrooms'],
                'values': [0.35, 0.28, 0.18, 0.12, 0.07]
            })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)