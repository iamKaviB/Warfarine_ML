# ============================================================================
# WARFARIN DOSE PREDICTION FLASK API
# ============================================================================

from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import logging
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('warfarin_api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# ============================================================================
# LOAD MODEL AND ARTIFACTS
# ============================================================================

try:
    model = joblib.load('warfarin_model.pkl')
    le_gender = joblib.load('gender_encoder.pkl')
    le_valve = joblib.load('valve_encoder.pkl')
    feature_names = joblib.load('feature_names.pkl')
    
    logger.info("✓ Model and artifacts loaded successfully")
    logger.info(f"  Model type: {type(model).__name__}")
    logger.info(f"  Features: {len(feature_names)}")
    
except Exception as e:
    logger.error(f"Failed to load model artifacts: {str(e)}")
    raise

# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

def validate_input(data):
    """Validate input data with clinical constraints"""
    errors = []
    
    # Required fields
    required_fields = ['age', 'gender', 'valve_position', 'current_dose', 
                       'current_inr', 'target_inr_min', 'target_inr_max']
    
    for field in required_fields:
        if field not in data:
            errors.append(f"Missing required field: {field}")
    
    if errors:
        return False, errors
    
    # Age validation
    age = data.get('age')
    if not isinstance(age, (int, float)) or age < 18 or age > 100:
        errors.append("Age must be between 18 and 100")
    
    # Gender validation
    gender = data.get('gender')
    valid_genders = list(le_gender.classes_)
    if gender not in valid_genders:
        errors.append(f"Gender must be one of: {valid_genders}")
    
    # Valve position validation
    valve_position = data.get('valve_position')
    valid_valves = list(le_valve.classes_)
    if valve_position not in valid_valves:
        errors.append(f"Valve position must be one of: {valid_valves}")
    
    # Current dose validation
    current_dose = data.get('current_dose')
    if not isinstance(current_dose, (int, float)) or current_dose < 0.5 or current_dose > 15:
        errors.append("Current dose must be between 0.5 and 15 mg/day")
    
    # INR validation
    current_inr = data.get('current_inr')
    if not isinstance(current_inr, (int, float)) or current_inr < 0.5 or current_inr > 8.0:
        errors.append("Current INR must be between 0.5 and 8.0")
    
    # Target INR validation
    target_min = data.get('target_inr_min')
    target_max = data.get('target_inr_max')
    
    if not isinstance(target_min, (int, float)) or target_min < 1.5 or target_min > 4.0:
        errors.append("Target INR minimum must be between 1.5 and 4.0")
    
    if not isinstance(target_max, (int, float)) or target_max < 2.0 or target_max > 4.5:
        errors.append("Target INR maximum must be between 2.0 and 4.5")
    
    if target_min >= target_max:
        errors.append("Target INR minimum must be less than maximum")
    
    return len(errors) == 0, errors

# ============================================================================
# FEATURE ENGINEERING FUNCTION
# ============================================================================

def create_prediction_features(data):
    """Create features for a single prediction"""
    
    # Encode categorical variables
    gender_enc = le_gender.transform([data['gender']])[0]
    valve_enc = le_valve.transform([data['valve_position']])[0]
    
    # Extract values
    age = data['age']
    current_dose = data['current_dose']
    current_inr = data['current_inr']
    target_min = data['target_inr_min']
    target_max = data['target_inr_max']
    
    # Calculate derived features
    target_mid = (target_min + target_max) / 2
    inr_deviation = current_inr - target_mid
    inr_below = 1 if current_inr < target_min else 0
    inr_above = 1 if current_inr > target_max else 0
    inr_in_range = 1 if (current_inr >= target_min and current_inr <= target_max) else 0
    inr_ratio = current_inr / target_mid
    inr_squared = current_inr ** 2
    
    # Age group: 0 (<45), 1 (45-60), 2 (>60)
    if age < 45:
        age_group = 0
    elif age <= 60:
        age_group = 1
    else:
        age_group = 2
    
    dose_per_inr = current_dose / current_inr
    dose_squared = current_dose ** 2
    
    # Create feature dictionary in correct order
    features = {
        'age': age,
        'gender_enc': gender_enc,
        'valve_position_enc': valve_enc,
        'current_dose': current_dose,
        'current_inr': current_inr,
        'target_inr_min': target_min,
        'target_inr_max': target_max,
        'target_inr_mid': target_mid,
        'inr_deviation_from_target': inr_deviation,
        'inr_below_range': inr_below,
        'inr_above_range': inr_above,
        'inr_in_range': inr_in_range,
        'inr_to_target_ratio': inr_ratio,
        'inr_squared': inr_squared,
        'age_group': age_group,
        'dose_per_inr': dose_per_inr,
        'dose_squared': dose_squared
    }
    
    # Convert to array in correct feature order
    feature_array = np.array([[features[f] for f in feature_names]])
    
    return feature_array, features

# ============================================================================
# CLINICAL CONSTRAINT APPLICATION
# ============================================================================

def apply_clinical_constraints(current_dose, predicted_dose, current_inr, 
                               target_inr_min, target_inr_max):
    """
    Apply clinical rules for dose adjustment
    
    Rules:
    1. Maximum change: ±0.5 mg from current dose
    2. Round to nearest 0.5 mg
    3. If INR below range: increase dose (max +0.5 mg)
    4. If INR above range: decrease dose (max -0.5 mg)
    5. If INR in range: use predicted dose with ±0.5 mg constraint
    """
    
    # Calculate raw dose change
    raw_change = predicted_dose - current_dose
    
    # Determine INR status
    inr_below = current_inr < target_inr_min
    inr_above = current_inr > target_inr_max
    inr_in_range = not (inr_below or inr_above)
    
    # Apply clinical logic
    if inr_below:
        # INR too low - MUST increase dose
        # If ML predicts increase, use it (capped at +0.5)
        # If ML predicts decrease or maintain, increase by +0.5
        if predicted_dose > current_dose:
            # ML wants to increase, cap at +0.5
            adjusted_dose = min(current_dose + 0.5, predicted_dose)
        else:
            # ML doesn't want to increase, but we must increase
            adjusted_dose = current_dose + 0.5
        reason = "INR below target range - dose increased"
        
    elif inr_above:
        # INR too high - MUST decrease dose
        # If ML predicts decrease, use it (capped at -0.5)
        # If ML predicts increase or maintain, decrease by -0.5
        if predicted_dose < current_dose:
            # ML wants to decrease, cap at -0.5
            adjusted_dose = max(current_dose - 0.5, predicted_dose)
        else:
            # ML doesn't want to decrease, but we must decrease
            adjusted_dose = current_dose - 0.5
        reason = "INR above target range - dose decreased"
        
    else:
        # INR in range - use predicted but cap at ±0.5
        if raw_change > 0.5:
            adjusted_dose = current_dose + 0.5
            reason = "INR in range - dose increase capped at +0.5 mg"
        elif raw_change < -0.5:
            adjusted_dose = current_dose - 0.5
            reason = "INR in range - dose decrease capped at -0.5 mg"
        else:
            adjusted_dose = predicted_dose
            reason = "INR in range - using ML predicted dose"
    
    # Round to nearest 0.5 mg
    adjusted_dose = round(adjusted_dose * 2) / 2
    
    # Ensure minimum dose of 0.5 mg
    adjusted_dose = max(0.5, adjusted_dose)
    
    # Determine dose change direction
    if adjusted_dose > current_dose:
        change_direction = "INCREASE"
    elif adjusted_dose < current_dose:
        change_direction = "DECREASE"
    else:
        change_direction = "MAINTAIN"
    
    return adjusted_dose, reason, change_direction
# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/predict', methods=['POST'])
def predict_dose():
    """
    Main prediction endpoint
    
    Expected JSON input:
    {
        "age": 60,
        "gender": "Male",
        "valve_position": "Mechanical Aortic",
        "current_dose": 5.5,
        "current_inr": 2.95,
        "target_inr_min": 2.0,
        "target_inr_max": 3.0
    }
    """
    
    try:
        # Get JSON data
        data = request.get_json()
        
        if not data:
            return jsonify({
                'error': 'No JSON data provided'
            }), 400
        
        # Log request
        logger.info(f"Prediction request received: {data}")
        
        # Validate input
        is_valid, errors = validate_input(data)
        if not is_valid:
            logger.warning(f"Validation failed: {errors}")
            return jsonify({
                'error': 'Validation failed',
                'details': errors
            }), 400
        
        # Create features
        feature_array, feature_dict = create_prediction_features(data)
        
        # Make prediction
        predicted_dose_raw = model.predict(feature_array)[0]
        
        # Apply clinical constraints
        adjusted_dose, reason, change_direction = apply_clinical_constraints(
            current_dose=data['current_dose'],
            predicted_dose=predicted_dose_raw,
            current_inr=data['current_inr'],
            target_inr_min=data['target_inr_min'],
            target_inr_max=data['target_inr_max']
        )
        
        # Calculate dose change
        dose_change = adjusted_dose - data['current_dose']
        dose_change_pct = (dose_change / data['current_dose']) * 100 if data['current_dose'] > 0 else 0
        
        # Prepare response
        response = {
            'success': True,
            'timestamp': datetime.now().isoformat(),
            'input': {
                'age': data['age'],
                'gender': data['gender'],
                'valve_position': data['valve_position'],
                'current_dose': data['current_dose'],
                'current_inr': data['current_inr'],
                'target_inr_range': f"{data['target_inr_min']}-{data['target_inr_max']}"
            },
            'prediction': {
                'ml_predicted_dose': round(predicted_dose_raw, 2),
                'recommended_dose': round(adjusted_dose, 2),
                'dose_change': round(dose_change, 2),
                'dose_change_percentage': round(dose_change_pct, 1),
                'change_direction': change_direction,
                'adjustment_reason': reason
            },
            'inr_status': {
                'below_range': feature_dict['inr_below_range'] == 1,
                'in_range': feature_dict['inr_in_range'] == 1,
                'above_range': feature_dict['inr_above_range'] == 1,
                'deviation_from_target': round(feature_dict['inr_deviation_from_target'], 2)
            },
            'clinical_notes': generate_clinical_notes(data, adjusted_dose, change_direction)
        }
        
        # Log successful prediction
        logger.info(f"Prediction successful: Current={data['current_dose']} mg, "
                   f"Predicted={predicted_dose_raw:.2f} mg, "
                   f"Recommended={adjusted_dose:.2f} mg")
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': 'Internal server error',
            'details': str(e)
        }), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """
    Batch prediction endpoint for multiple patients
    
    Expected JSON input:
    {
        "patients": [
            {
                "patient_id": "P001",
                "age": 60,
                ...
            },
            ...
        ]
    }
    """
    
    try:
        data = request.get_json()
        
        if not data or 'patients' not in data:
            return jsonify({
                'error': 'No patient data provided'
            }), 400
        
        patients = data['patients']
        results = []
        
        for i, patient in enumerate(patients):
            try:
                # Add patient to request and get prediction
                feature_array, feature_dict = create_prediction_features(patient)
                predicted_dose_raw = model.predict(feature_array)[0]
                
                adjusted_dose, reason, change_direction = apply_clinical_constraints(
                    current_dose=patient['current_dose'],
                    predicted_dose=predicted_dose_raw,
                    current_inr=patient['current_inr'],
                    target_inr_min=patient['target_inr_min'],
                    target_inr_max=patient['target_inr_max']
                )
                
                results.append({
                    'patient_id': patient.get('patient_id', f'P{i+1}'),
                    'success': True,
                    'recommended_dose': round(adjusted_dose, 2),
                    'change_direction': change_direction
                })
                
            except Exception as e:
                results.append({
                    'patient_id': patient.get('patient_id', f'P{i+1}'),
                    'success': False,
                    'error': str(e)
                })
        
        return jsonify({
            'success': True,
            'total_patients': len(patients),
            'results': results
        }), 200
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'details': str(e)
        }), 500

@app.route('/model_info', methods=['GET'])
def model_info():
    """Get information about the loaded model"""
    return jsonify({
        'model_type': type(model).__name__,
        'n_features': len(feature_names),
        'features': feature_names,
        'valid_genders': list(le_gender.classes_),
        'valid_valve_positions': list(le_valve.classes_),
        'constraints': {
            'max_dose_change': 0.5,
            'dose_rounding': 0.5,
            'min_dose': 0.5,
            'max_dose': 15.0,
            'inr_range': '0.5-8.0'
        }
    })

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def generate_clinical_notes(data, recommended_dose, change_direction):
    """Generate clinical notes for the recommendation"""
    notes = []
    
    current_inr = data['current_inr']
    target_min = data['target_inr_min']
    target_max = data['target_inr_max']
    
    # INR status note
    if current_inr < target_min:
        notes.append(f"Current INR ({current_inr}) is below target range ({target_min}-{target_max}). "
                    f"Increase in dose recommended.")
    elif current_inr > target_max:
        notes.append(f"Current INR ({current_inr}) is above target range ({target_min}-{target_max}). "
                    f"Decrease in dose recommended.")
    else:
        notes.append(f"Current INR ({current_inr}) is within target range ({target_min}-{target_max}).")
    
    # Dose change note
    if change_direction == "MAINTAIN":
        notes.append("Current dose is appropriate. No change recommended.")
    else:
        notes.append(f"Dose change limited to maximum ±0.5 mg per clinical guidelines.")
    
    # Follow-up note
    notes.append("Recommend repeat INR testing in 1-2 weeks to assess response.")
    
    return notes

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    logger.info("Starting Warfarin Dose Prediction API")
    logger.info(f"Model: {type(model).__name__}")
    logger.info(f"Features: {len(feature_names)}")
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False  # Set to False in production
    )