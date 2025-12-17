from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)

# Load model and encoders at startup
model = joblib.load('inr_prediction_model.pkl')
label_encoders = joblib.load('label_encoders.pkl')
le_target = joblib.load('target_encoder.pkl')
features = joblib.load('feature_names.pkl')
feature_importance = joblib.load('feature_importance.pkl')

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': True,
        'model_type': 'RandomForestClassifier',
        'n_features': len(features),
        'n_classes': len(le_target.classes_)
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data
        data = request.get_json()
        
        # Validate required fields
        missing_features = [f for f in features if f not in data]
        if missing_features:
            return jsonify({
                'error': 'Missing required features',
                'missing_features': missing_features
            }), 400
        
        # Create dataframe from input
        input_df = pd.DataFrame([data])
        
        # Encode input data
        for col in features:
            if col in label_encoders:
                try:
                    input_df[col] = label_encoders[col].transform(input_df[col].astype(str))
                except:
                    # Handle unseen categories
                    input_df[col] = 0
        
        # Make prediction
        prediction = model.predict(input_df[features])[0]
        probabilities = model.predict_proba(input_df[features])[0]
        
        predicted_status = le_target.inverse_transform([prediction])[0]
        confidence = float(probabilities[prediction] * 100)
        
        # Get all class probabilities
        all_probabilities = {}
        for i, status_class in enumerate(le_target.classes_):
            all_probabilities[status_class] = float(probabilities[i] * 100)
        
        # Calculate feature contributions
        feature_contributions = {}
        
        for i, feature in enumerate(features):
            feature_value = input_df[feature].values[0]
            importance = feature_importance[feature_importance['Feature'] == feature]['Importance'].values[0]
            
            # Decode the actual value
            if feature in label_encoders:
                try:
                    actual_value = label_encoders[feature].inverse_transform([int(feature_value)])[0]
                except:
                    actual_value = data.get(feature, 'Unknown')
            else:
                actual_value = feature_value
                
            feature_contributions[feature] = {
                'value': str(actual_value),
                'importance': float(importance),
                'contribution_score': float(importance * (feature_value + 1))
            }
        
        # Sort by contribution score
        sorted_contributions = sorted(
            feature_contributions.items(), 
            key=lambda x: x[1]['contribution_score'], 
            reverse=True
        )
        
        # Get top 5 reasons
        top_reasons = []
        for feature, info in sorted_contributions[:5]:
            impact = 'High' if info['importance'] > 0.1 else 'Moderate' if info['importance'] > 0.05 else 'Low'
            top_reasons.append({
                'factor': feature,
                'value': info['value'],
                'importance': f"{info['importance']:.4f}",
                'impact': impact
            })
        
        # Risk assessment
        if confidence > 80:
            risk_level = "High Confidence"
        elif confidence > 60:
            risk_level = "Moderate Confidence"
        else:
            risk_level = "Low Confidence - Recommend Clinical Review"
        
        # Generate recommendations
        recommendations = generate_recommendations(predicted_status, top_reasons, data)
        
        # Return response
        return jsonify({
            'success': True,
            'predicted_status': predicted_status,
            'confidence': f"{confidence:.2f}%",
            'confidence_score': round(confidence, 2),
            'all_probabilities': all_probabilities,
            'top_contributing_factors': top_reasons,
            'risk_level': risk_level,
            'recommendations': recommendations
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

def generate_recommendations(predicted_status, top_reasons, input_data):
    recommendations = []
    
    # Status-specific recommendations
    if 'low' in predicted_status.lower():
        recommendations.append("⚠️ Monitor for signs of bleeding or bruising")
        recommendations.append("📋 Consider dose adjustment - consult healthcare provider")
    elif 'high' in predicted_status.lower() or 'increase' in predicted_status.lower():
        recommendations.append("⚠️ Monitor for signs of clotting")
        recommendations.append("📋 Review recent dietary changes and medication compliance")
    elif 'in range' in predicted_status.lower():
        recommendations.append("✅ Continue current regimen")
        recommendations.append("📋 Maintain consistent diet and medication schedule")
    
    # Factor-specific recommendations
    for reason in top_reasons:
        factor = reason['factor']
        value = reason['value']
        
        if 'Missed Dose' in factor and value not in ['Never', 'No', 'nan']:
            recommendations.append("💊 Set medication reminders to improve compliance")
        
        if 'Vitamin K' in factor and value == 'Yes':
            recommendations.append("🥗 Maintain consistent Vitamin K intake")
        
        if 'dietary changes' in factor.lower() and value == 'Yes':
            recommendations.append("🍽️ Gradual dietary changes preferred - inform healthcare provider")
        
        if 'antibiotics' in factor.lower() and value == 'Yes':
            recommendations.append("💊 Antibiotics can affect INR - closer monitoring needed")
        
        if 'extra doses' in factor.lower() and value == 'Yes':
            recommendations.append("⚠️ Never take extra doses without consulting doctor")
    
    return list(set(recommendations))

@app.route('/model-info', methods=['GET'])
def model_info():
    return jsonify({
        'model_type': 'RandomForestClassifier',
        'n_estimators': 200,
        'max_depth': 10,
        'features': features,
        'n_features': len(features),
        'target_classes': le_target.classes_.tolist(),
        'n_classes': len(le_target.classes_),
        'feature_importance': feature_importance.to_dict('records')
    })

if __name__ == '__main__':
    print("Starting INR Prediction API...")
    print(f"Model loaded: {model is not None}")
    print(f"Features: {len(features)}")
    print(f"Classes: {len(le_target.classes_)}")
    app.run(host='0.0.0.0', port=6000, debug=True)
