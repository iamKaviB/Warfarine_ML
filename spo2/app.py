"""
SpO2 Patient Label Prediction REST API
Complete REST API with monthly aggregation and batch processing
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import logging
from datetime import datetime
from functools import wraps
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
NUM_DAYS = 29
API_VERSION = "1.0.0"
MODEL_FILES = {
    'model': 'spo2_model.pkl',
    'scaler': 'scaler.pkl',
    'label_encoder': 'label_encoder.pkl',
    'feature_columns': 'feature_columns.pkl'
}

# Global variables for loaded models
model = None
scaler = None
label_encoder = None
feature_columns = None
model_loaded = False

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_models():
    """Load all model files"""
    global model, scaler, label_encoder, feature_columns, model_loaded
    
    try:
        model = joblib.load(MODEL_FILES['model'])
        scaler = joblib.load(MODEL_FILES['scaler'])
        label_encoder = joblib.load(MODEL_FILES['label_encoder'])
        feature_columns = joblib.load(MODEL_FILES['feature_columns'])
        model_loaded = True
        logger.info("✓ All models loaded successfully")
        return True
    except FileNotFoundError as e:
        logger.error(f"✗ Model file not found: {e}")
        model_loaded = False
        return False
    except Exception as e:
        logger.error(f"✗ Error loading models: {e}")
        model_loaded = False
        return False

def require_model(f):
    """Decorator to ensure model is loaded before processing"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not model_loaded:
            return jsonify({
                'error': 'Model not loaded',
                'message': 'Please ensure all model files are present',
                'required_files': list(MODEL_FILES.values())
            }), 503
        return f(*args, **kwargs)
    return decorated_function

def validate_spo2_data(data, required_days=NUM_DAYS):
    """Validate SpO2 input data"""
    errors = []
    
    # Check required fields
    required_keys = [f'val_{i}' for i in range(1, required_days + 1)]
    missing_keys = [key for key in required_keys if key not in data]
    
    if missing_keys:
        errors.append(f"Missing {len(missing_keys)} day(s) of data: {missing_keys[:5]}...")
    
    # Validate values
    for key in required_keys:
        if key in data:
            value = data[key]
            if not isinstance(value, (int, float)):
                errors.append(f"{key} must be a number")
            elif value < 0 or value > 100:
                errors.append(f"{key} must be between 0 and 100, got {value}")
    
    return errors

def create_features_from_input(data):
    """Create features from raw input data (29 days)"""
    df = pd.DataFrame([data])
    spo2_cols = [f'val_{i}' for i in range(1, NUM_DAYS + 1)]
    
    features = {}
    
    # Monthly statistics (29 days)
    features['monthly_mean'] = df[spo2_cols].mean(axis=1).values[0]
    features['monthly_std'] = df[spo2_cols].std(axis=1).values[0]
    features['monthly_min'] = df[spo2_cols].min(axis=1).values[0]
    features['monthly_max'] = df[spo2_cols].max(axis=1).values[0]
    features['monthly_median'] = df[spo2_cols].median(axis=1).values[0]
    features['monthly_range'] = features['monthly_max'] - features['monthly_min']
    
    # Weekly statistics (4 weeks)
    for week in range(4):
        start_day = week * 7 + 1
        end_day = min(start_day + 7, NUM_DAYS + 1)
        week_cols = [f'val_{i}' for i in range(start_day, end_day)]
        
        features[f'week{week+1}_mean'] = df[week_cols].mean(axis=1).values[0]
        features[f'week{week+1}_std'] = df[week_cols].std(axis=1).values[0]
        features[f'week{week+1}_min'] = df[week_cols].min(axis=1).values[0]
        features[f'week{week+1}_max'] = df[week_cols].max(axis=1).values[0]
    
    # Trend features
    first_week_cols = [f'val_{i}' for i in range(1, 8)]
    last_week_cols = [f'val_{i}' for i in range(23, NUM_DAYS + 1)]
    features['first_week_mean'] = df[first_week_cols].mean(axis=1).values[0]
    features['last_week_mean'] = df[last_week_cols].mean(axis=1).values[0]
    features['trend'] = features['last_week_mean'] - features['first_week_mean']
    
    # Count features
    features['days_below_90'] = (df[spo2_cols] < 90).sum(axis=1).values[0]
    features['days_below_95'] = (df[spo2_cols] < 95).sum(axis=1).values[0]
    features['days_above_97'] = (df[spo2_cols] > 97).sum(axis=1).values[0]
    
    # Volatility
    features['volatility'] = df[spo2_cols].apply(lambda x: x.diff().abs().mean(), axis=1).values[0]
    
    return features

def calculate_monthly_stats(data):
    """Calculate comprehensive monthly statistics"""
    spo2_cols = [f'val_{i}' for i in range(1, NUM_DAYS + 1)]
    values = [data[col] for col in spo2_cols]
    
    stats = {
        'mean': float(np.mean(values)),
        'median': float(np.median(values)),
        'std': float(np.std(values)),
        'min': float(np.min(values)),
        'max': float(np.max(values)),
        'range': float(np.max(values) - np.min(values)),
        'q1': float(np.percentile(values, 25)),
        'q3': float(np.percentile(values, 75)),
        'days_below_90': int(sum(1 for v in values if v < 90)),
        'days_below_95': int(sum(1 for v in values if v < 95)),
        'days_above_97': int(sum(1 for v in values if v > 97)),
        'total_days': NUM_DAYS
    }
    
    # Add percentage metrics
    stats['percent_below_90'] = (stats['days_below_90'] / NUM_DAYS) * 100
    stats['percent_below_95'] = (stats['days_below_95'] / NUM_DAYS) * 100
    stats['percent_above_97'] = (stats['days_above_97'] / NUM_DAYS) * 100
    
    # Risk assessment based on monthly stats
    if stats['mean'] < 90 or stats['days_below_90'] > NUM_DAYS * 0.3:
        stats['risk_level'] = 'high'
    elif stats['mean'] < 95 or stats['days_below_95'] > NUM_DAYS * 0.5:
        stats['risk_level'] = 'medium'
    else:
        stats['risk_level'] = 'low'
    
    return stats

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/', methods=['GET'])
def index():
    """API root endpoint"""
    return jsonify({
        'name': 'SpO2 Patient Label Prediction API',
        'version': API_VERSION,
        'status': 'running',
        'model_loaded': model_loaded,
        'endpoints': {
            'GET /': 'API information',
            'GET /health': 'Health check',
            'GET /api/info': 'Detailed API information',
            'POST /api/predict': 'Single patient prediction',
            'POST /api/predict/batch': 'Batch predictions',
            'POST /api/monthly/stats': 'Monthly statistics only',
            'POST /api/monthly/analyze': 'Monthly analysis with prediction',
            'POST /api/validate': 'Validate SpO2 data'
        },
        'documentation': 'See README for detailed API documentation'
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    health_status = {
        'status': 'healthy' if model_loaded else 'unhealthy',
        'model_loaded': model_loaded,
        'timestamp': datetime.now().isoformat(),
        'uptime': 'running'
    }
    
    if model_loaded:
        health_status['model_info'] = {
            'features': len(feature_columns),
            'labels': list(label_encoder.classes_)
        }
    
    status_code = 200 if model_loaded else 503
    return jsonify(health_status), status_code

@app.route('/api/info', methods=['GET'])
def api_info():
    """Detailed API information"""
    return jsonify({
        'api_version': API_VERSION,
        'model_loaded': model_loaded,
        'configuration': {
            'num_days': NUM_DAYS,
            'features_count': len(feature_columns) if feature_columns else 0,
            'labels': list(label_encoder.classes_) if label_encoder else []
        },
        'input_format': {
            'required_fields': [f'val_{i}' for i in range(1, NUM_DAYS + 1)],
            'value_range': [0, 100],
            'data_type': 'integer or float'
        },
        'output_format': {
            'prediction': 'string (bad/okay/warning)',
            'confidence': 'float (0.0 to 1.0)',
            'probabilities': 'object with all class probabilities',
            'monthly_stats': 'object with statistical analysis'
        }
    })

@app.route('/api/validate', methods=['POST'])
def validate_data():
    """Validate SpO2 data without making prediction"""
    try:
        data = request.json
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Validate
        errors = validate_spo2_data(data)
        
        if errors:
            return jsonify({
                'valid': False,
                'errors': errors,
                'error_count': len(errors)
            }), 400
        
        # Calculate basic stats
        monthly_stats = calculate_monthly_stats(data)
        
        return jsonify({
            'valid': True,
            'message': 'Data is valid',
            'monthly_stats': monthly_stats
        })
        
    except Exception as e:
        logger.error(f"Validation error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict', methods=['POST'])
@require_model
def predict():
    """Single patient prediction"""
    try:
        data = request.json
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Validate input
        errors = validate_spo2_data(data)
        if errors:
            return jsonify({
                'error': 'Invalid input data',
                'validation_errors': errors
            }), 400
        
        # Create features
        features = create_features_from_input(data)
        
        # Create feature vector
        X = np.array([[features[col] for col in feature_columns]])
        X_scaled = scaler.transform(X)
        
        # Make prediction
        prediction = model.predict(X_scaled)[0]
        prediction_proba = model.predict_proba(X_scaled)[0]
        
        # Get label and confidence
        label = label_encoder.inverse_transform([prediction])[0]
        confidence = float(prediction_proba[prediction])
        
        # Calculate monthly statistics
        monthly_stats = calculate_monthly_stats(data)
        
        # Prepare response
        response = {
            'prediction': label,
            'confidence': confidence,
            'probabilities': {
                label_encoder.classes_[i]: float(prob)
                for i, prob in enumerate(prediction_proba)
            },
            'monthly_stats': monthly_stats,
            'features': {
                'monthly_mean': float(features['monthly_mean']),
                'monthly_std': float(features['monthly_std']),
                'trend': float(features['trend']),
                'volatility': float(features['volatility'])
            },
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Prediction: {label} (confidence: {confidence:.3f})")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict/batch', methods=['POST'])
@require_model
def predict_batch():
    """Batch predictions for multiple patients"""
    try:
        data = request.json
        
        if not data or 'patients' not in data:
            return jsonify({
                'error': 'Invalid format. Expected: {"patients": [patient1, patient2, ...]}'
            }), 400
        
        patients = data['patients']
        
        if not isinstance(patients, list):
            return jsonify({'error': 'patients must be an array'}), 400
        
        if len(patients) == 0:
            return jsonify({'error': 'No patients provided'}), 400
        
        if len(patients) > 100:
            return jsonify({'error': 'Maximum 100 patients per batch'}), 400
        
        results = []
        errors = []
        
        for idx, patient_data in enumerate(patients):
            try:
                # Validate
                validation_errors = validate_spo2_data(patient_data)
                if validation_errors:
                    errors.append({
                        'patient_index': idx,
                        'errors': validation_errors
                    })
                    continue
                
                # Create features
                features = create_features_from_input(patient_data)
                X = np.array([[features[col] for col in feature_columns]])
                X_scaled = scaler.transform(X)
                
                # Predict
                prediction = model.predict(X_scaled)[0]
                prediction_proba = model.predict_proba(X_scaled)[0]
                label = label_encoder.inverse_transform([prediction])[0]
                
                # Monthly stats
                monthly_stats = calculate_monthly_stats(patient_data)
                
                results.append({
                    'patient_index': idx,
                    'prediction': label,
                    'confidence': float(prediction_proba[prediction]),
                    'monthly_stats': monthly_stats
                })
                
            except Exception as e:
                errors.append({
                    'patient_index': idx,
                    'error': str(e)
                })
        
        return jsonify({
            'success_count': len(results),
            'error_count': len(errors),
            'results': results,
            'errors': errors if errors else None,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/monthly/stats', methods=['POST'])
def monthly_stats():
    """Get monthly statistics without prediction"""
    try:
        data = request.json
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Validate
        errors = validate_spo2_data(data)
        if errors:
            return jsonify({
                'error': 'Invalid input data',
                'validation_errors': errors
            }), 400
        
        # Calculate comprehensive stats
        stats = calculate_monthly_stats(data)
        
        # Add weekly breakdown
        weekly_stats = []
        for week in range(4):
            start_day = week * 7 + 1
            end_day = min(start_day + 7, NUM_DAYS + 1)
            week_values = [data[f'val_{i}'] for i in range(start_day, end_day)]
            
            weekly_stats.append({
                'week': week + 1,
                'days': f"{start_day}-{end_day-1}",
                'mean': float(np.mean(week_values)),
                'min': float(np.min(week_values)),
                'max': float(np.max(week_values))
            })
        
        return jsonify({
            'monthly_stats': stats,
            'weekly_breakdown': weekly_stats,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Monthly stats error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/monthly/analyze', methods=['POST'])
@require_model
def monthly_analyze():
    """Complete monthly analysis with prediction and detailed insights"""
    try:
        data = request.json
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Get patient ID if provided
        patient_id = data.pop('patient_id', None)
        month = data.pop('month', None)
        year = data.pop('year', None)
        
        # Validate
        errors = validate_spo2_data(data)
        if errors:
            return jsonify({
                'error': 'Invalid input data',
                'validation_errors': errors
            }), 400
        
        # Create features
        features = create_features_from_input(data)
        X = np.array([[features[col] for col in feature_columns]])
        X_scaled = scaler.transform(X)
        
        # Predict
        prediction = model.predict(X_scaled)[0]
        prediction_proba = model.predict_proba(X_scaled)[0]
        label = label_encoder.inverse_transform([prediction])[0]
        
        # Monthly stats
        monthly_stats = calculate_monthly_stats(data)
        
        # Weekly breakdown
        weekly_analysis = []
        for week in range(4):
            start_day = week * 7 + 1
            end_day = min(start_day + 7, NUM_DAYS + 1)
            week_values = [data[f'val_{i}'] for i in range(start_day, end_day)]
            
            weekly_analysis.append({
                'week': week + 1,
                'days_range': f"Day {start_day}-{end_day-1}",
                'mean': float(np.mean(week_values)),
                'min': float(np.min(week_values)),
                'max': float(np.max(week_values)),
                'std': float(np.std(week_values)),
                'days_below_90': int(sum(1 for v in week_values if v < 90))
            })
        
        # Generate insights
        insights = []
        
        # Check for prediction-stats mismatch
        if label == 'bad' and monthly_stats['risk_level'] == 'low':
            insights.append("⚠️ ALERT: Model predicted 'bad' but monthly statistics indicate low risk")
            insights.append("This may indicate specific patterns or deteriorating trend that require attention")
        elif label == 'okay' and monthly_stats['risk_level'] == 'high':
            insights.append("⚠️ ALERT: Model predicted 'okay' but monthly statistics indicate high risk")
            insights.append("Review individual daily patterns for concerning variations")
        elif label == 'warning' and monthly_stats['risk_level'] == 'low':
            insights.append("⚠️ Note: Model predicted 'warning' despite low-risk monthly statistics")
        
        # Monthly average assessment
        if monthly_stats['mean'] < 90:
            insights.append("Critical: Monthly average SpO2 is below 90%")
        elif monthly_stats['mean'] < 95:
            insights.append("Warning: Monthly average SpO2 is below 95%")
        else:
            insights.append("Good: Monthly average SpO2 is within normal range")
        
        # Daily episodes
        if monthly_stats['days_below_90'] > 5:
            insights.append(f"Serious concern: {monthly_stats['days_below_90']} day(s) with SpO2 below 90%")
        elif monthly_stats['days_below_90'] > 0:
            insights.append(f"Alert: {monthly_stats['days_below_90']} day(s) with SpO2 below 90%")
        
        # Trend analysis
        if features['trend'] < -5:
            insights.append(f"Significant declining trend: SpO2 decreased by {abs(features['trend']):.1f}% over the month")
        elif features['trend'] < -2:
            insights.append("Declining trend: SpO2 decreased over the month")
        elif features['trend'] > 5:
            insights.append(f"Improving trend: SpO2 increased by {features['trend']:.1f}% over the month")
        elif features['trend'] > 2:
            insights.append("Improving trend: SpO2 increased over the month")
        
        # Volatility
        if features['volatility'] > 5:
            insights.append("Very high volatility: SpO2 levels are highly unstable")
        elif features['volatility'] > 3:
            insights.append("High volatility: SpO2 levels are unstable")
        
        # Prepare response
        response = {
            'patient_info': {
                'patient_id': patient_id,
                'month': month,
                'year': year,
                'analysis_date': datetime.now().isoformat()
            },
            'prediction': {
                'label': label,
                'confidence': float(prediction_proba[prediction]),
                'probabilities': {
                    label_encoder.classes_[i]: float(prob)
                    for i, prob in enumerate(prediction_proba)
                }
            },
            'monthly_summary': monthly_stats,
            'weekly_breakdown': weekly_analysis
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Monthly analysis error: {str(e)}")
        return jsonify({'error': str(e)}), 500

def generate_recommendations(label, stats):
    """Generate recommendations based on prediction and stats"""
    recommendations = []
    
    # Base recommendations on actual label
    if label == 'bad':
        recommendations.extend([
            "Immediate medical attention recommended",
            "Monitor SpO2 levels continuously",
            "Consider oxygen therapy if not already administered"
        ])
    elif label == 'warning':
        recommendations.extend([
            "Increased monitoring recommended",
            "Consult healthcare provider for assessment",
            "Track daily SpO2 trends closely"
        ])
    else:  # okay
        recommendations.extend([
            "Continue current health management",
            "Maintain regular monitoring schedule"
        ])
    
    # Add specific recommendations based on actual stats
    if stats['days_below_90'] > 5:
        recommendations.append("Multiple days below 90% detected - schedule medical evaluation")
    elif stats['days_below_90'] > 0 and label != 'bad':
        recommendations.append(f"{stats['days_below_90']} day(s) below 90% - monitor closely")
    
    if stats['risk_level'] == 'high':
        recommendations.append("High risk level detected - immediate action required")
    elif stats['risk_level'] == 'medium' and label == 'okay':
        recommendations.append("Medium risk level - increased vigilance recommended")
    
    # Check for mismatches and add clarifying note
    if label == 'bad' and stats['risk_level'] == 'low':
        recommendations.insert(0, "Note: Model predicted 'bad' but stats show low risk - review individual patterns")
    elif label == 'okay' and stats['risk_level'] == 'high':
        recommendations.insert(0, "Note: Model predicted 'okay' but stats show high risk - seek medical advice")
    
    return recommendations

# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'error': 'Endpoint not found',
        'message': 'The requested endpoint does not exist',
        'available_endpoints': [
            'GET /',
            'GET /health',
            'GET /api/info',
            'POST /api/predict',
            'POST /api/predict/batch',
            'POST /api/monthly/stats',
            'POST /api/monthly/analyze',
            'POST /api/validate'
        ]
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'error': 'Internal server error',
        'message': 'An unexpected error occurred'
    }), 500

# ============================================================================
# STARTUP
# ============================================================================

if __name__ == '__main__':
    print("=" * 80)
    print("SpO2 Patient Label Prediction REST API")
    print("=" * 80)
    print(f"Version: {API_VERSION}")
    print(f"Days of data: {NUM_DAYS}")
    print()
    
    # Load models
    print("Loading models...")
    if load_models():
        print("✓ Models loaded successfully")
        print(f"✓ Features: {len(feature_columns)}")
        print(f"✓ Labels: {list(label_encoder.classes_)}")
    else:
        print("✗ Failed to load models")
        print("⚠  API will start but predictions will not be available")
    
    print()
    print("Starting server...")
    print("API will be available at: http://localhost:5000")
    print("=" * 80)
    print()
    
    # Use port 5000 (default) or 8080 if 5000 is occupied by AirPlay
    PORT = int(os.environ.get('PORT', 5001))
    app.run(debug=True, host='0.0.0.0', port=PORT)