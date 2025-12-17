# ============================================================================
# WARFARIN API TESTING SCRIPT
# ============================================================================

import requests
import json
import pandas as pd
from datetime import datetime

# API Configuration
API_BASE_URL = "http://localhost:5000"

# ============================================================================
# TEST CASES
# ============================================================================

def test_health_check():
    """Test the health check endpoint"""
    print("\n" + "="*80)
    print("TEST 1: Health Check")
    print("="*80)
    
    response = requests.get(f"{API_BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    return response.status_code == 200

def test_model_info():
    """Test the model info endpoint"""
    print("\n" + "="*80)
    print("TEST 2: Model Information")
    print("="*80)
    
    response = requests.get(f"{API_BASE_URL}/model_info")
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"\nModel Type: {data['model_type']}")
        print(f"Number of Features: {data['n_features']}")
        print(f"\nValid Genders: {data['valid_genders']}")
        print(f"Valid Valve Positions: {data['valid_valve_positions']}")
        print(f"\nConstraints:")
        for key, value in data['constraints'].items():
            print(f"  {key}: {value}")
    
    return response.status_code == 200

def test_single_prediction():
    """Test single dose prediction"""
    print("\n" + "="*80)
    print("TEST 3: Single Prediction - INR Below Range")
    print("="*80)
    
    # Patient with INR below target range
    patient_data = {
        "age": 60,
        "gender": "Female",
        "valve_position": "Mechanical Aortic",
        "current_dose": 5.5,
        "current_inr": 1.8,  # Below target
        "target_inr_min": 2.0,
        "target_inr_max": 3.0
    }
    
    print(f"Input Data:")
    print(json.dumps(patient_data, indent=2))
    
    response = requests.post(
        f"{API_BASE_URL}/predict",
        json=patient_data,
        headers={'Content-Type': 'application/json'}
    )
    
    print(f"\nStatus Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"\n{'='*80}")
        print("PREDICTION RESULT:")
        print(f"{'='*80}")
        print(f"Current Dose: {result['input']['current_dose']} mg/day")
        print(f"Current INR: {result['input']['current_inr']}")
        print(f"Target INR Range: {result['input']['target_inr_range']}")
        print(f"\nML Predicted Dose: {result['prediction']['ml_predicted_dose']} mg/day")
        print(f"Recommended Dose: {result['prediction']['recommended_dose']} mg/day")
        print(f"Dose Change: {result['prediction']['dose_change']:+.1f} mg ({result['prediction']['change_direction']})")
        print(f"Adjustment Reason: {result['prediction']['adjustment_reason']}")
        print(f"\nINR Status:")
        print(f"  Below Range: {result['inr_status']['below_range']}")
        print(f"  In Range: {result['inr_status']['in_range']}")
        print(f"  Above Range: {result['inr_status']['above_range']}")
        print(f"\nClinical Notes:")
        for note in result['clinical_notes']:
            print(f"  • {note}")
    else:
        print(f"Error: {response.json()}")
    
    return response.status_code == 200

def test_inr_above_range():
    """Test prediction with INR above range"""
    print("\n" + "="*80)
    print("TEST 4: Single Prediction - INR Above Range")
    print("="*80)
    
    patient_data = {
        "age": 47,
        "gender": "Men",
        "valve_position": "Mechanical Mitral",
        "current_dose": 4.0,
        "current_inr": 3.91,  # Above target
        "target_inr_min": 2.5,
        "target_inr_max": 3.5
    }
    
    print(f"Input Data:")
    print(json.dumps(patient_data, indent=2))
    
    response = requests.post(
        f"{API_BASE_URL}/predict",
        json=patient_data,
        headers={'Content-Type': 'application/json'}
    )
    
    print(f"\nStatus Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"\nRecommended Dose: {result['prediction']['recommended_dose']} mg/day")
        print(f"Dose Change: {result['prediction']['dose_change']:+.1f} mg ({result['prediction']['change_direction']})")
        print(f"Reason: {result['prediction']['adjustment_reason']}")
    
    return response.status_code == 200

def test_inr_in_range():
    """Test prediction with INR in target range"""
    print("\n" + "="*80)
    print("TEST 5: Single Prediction - INR In Range")
    print("="*80)
    
    patient_data = {
        "age": 69,
        "gender": "Female",
        "valve_position": "Mechanical Aortic",
        "current_dose": 3.5,
        "current_inr": 2.2,  # In target range
        "target_inr_min": 2.0,
        "target_inr_max": 3.0
    }
    
    print(f"Input Data:")
    print(json.dumps(patient_data, indent=2))
    
    response = requests.post(
        f"{API_BASE_URL}/predict",
        json=patient_data,
        headers={'Content-Type': 'application/json'}
    )
    
    print(f"\nStatus Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"\nRecommended Dose: {result['prediction']['recommended_dose']} mg/day")
        print(f"Dose Change: {result['prediction']['dose_change']:+.1f} mg ({result['prediction']['change_direction']})")
        print(f"Reason: {result['prediction']['adjustment_reason']}")
    
    return response.status_code == 200

def test_batch_prediction():
    """Test batch prediction endpoint"""
    print("\n" + "="*80)
    print("TEST 6: Batch Prediction")
    print("="*80)
    
    batch_data = {
        "patients": [
            {
                "patient_id": "P001",
                "age": 60,
                "gender": "Female",
                "valve_position": "Mechanical Aortic",
                "current_dose": 5.5,
                "current_inr": 2.95,
                "target_inr_min": 2.0,
                "target_inr_max": 3.0
            },
            {
                "patient_id": "P002",
                "age": 47,
                "gender": "Men",
                "valve_position": "Mechanical Mitral",
                "current_dose": 4.0,
                "current_inr": 2.63,
                "target_inr_min": 2.5,
                "target_inr_max": 3.5
            },
            {
                "patient_id": "P003",
                "age": 50,
                "gender": "Female",
                "valve_position": "Mechanical Mitral",
                "current_dose": 4.0,
                "current_inr": 3.25,
                "target_inr_min": 2.5,
                "target_inr_max": 3.5
            }
        ]
    }
    
    response = requests.post(
        f"{API_BASE_URL}/batch_predict",
        json=batch_data,
        headers={'Content-Type': 'application/json'}
    )
    
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"\nTotal Patients: {result['total_patients']}")
        print(f"\nResults:")
        
        df = pd.DataFrame(result['results'])
        print(df.to_string(index=False))
    
    return response.status_code == 200

def test_validation_errors():
    """Test input validation"""
    print("\n" + "="*80)
    print("TEST 7: Input Validation Errors")
    print("="*80)
    
    # Test 1: Missing required field
    print("\nTest 7a: Missing required field")
    invalid_data = {
        "age": 60,
        "gender": "Female",
        # Missing valve_position
        "current_dose": 5.5,
        "current_inr": 2.95
    }
    
    response = requests.post(
        f"{API_BASE_URL}/predict",
        json=invalid_data,
        headers={'Content-Type': 'application/json'}
    )
    
    print(f"Status Code: {response.status_code}")
    if response.status_code == 400:
        print(f"Error (Expected): {response.json()['error']}")
        print("✓ Validation working correctly")
    
    # Test 2: Invalid age
    print("\nTest 7b: Invalid age value")
    invalid_data = {
        "age": 150,  # Invalid
        "gender": "Female",
        "valve_position": "Mechanical Aortic",
        "current_dose": 5.5,
        "current_inr": 2.95,
        "target_inr_min": 2.0,
        "target_inr_max": 3.0
    }
    
    response = requests.post(
        f"{API_BASE_URL}/predict",
        json=invalid_data,
        headers={'Content-Type': 'application/json'}
    )
    
    print(f"Status Code: {response.status_code}")
    if response.status_code == 400:
        print(f"Error (Expected): {response.json()['details']}")
        print("✓ Validation working correctly")
    
    # Test 3: Invalid INR
    print("\nTest 7c: Invalid INR value")
    invalid_data = {
        "age": 60,
        "gender": "Female",
        "valve_position": "Mechanical Aortic",
        "current_dose": 5.5,
        "current_inr": 10.0,  # Invalid
        "target_inr_min": 2.0,
        "target_inr_max": 3.0
    }
    
    response = requests.post(
        f"{API_BASE_URL}/predict",
        json=invalid_data,
        headers={'Content-Type': 'application/json'}
    )
    
    print(f"Status Code: {response.status_code}")
    if response.status_code == 400:
        print(f"Error (Expected): {response.json()['details']}")
        print("✓ Validation working correctly")
    
    return True

def test_real_dataset():
    """Test with real dataset examples"""
    print("\n" + "="*80)
    print("TEST 8: Real Dataset Examples")
    print("="*80)
    
    # Real examples from the dataset
    real_patients = [
        {
            "name": "Patient 1 - Female, 69, Aortic Valve",
            "data": {
                "age": 69,
                "gender": "Female",
                "valve_position": "Mechanical Aortic",
                "current_dose": 3.5,
                "current_inr": 2.2,
                "target_inr_min": 2.0,
                "target_inr_max": 3.0
            },
            "expected_dose": 3.5
        },
        {
            "name": "Patient 2 - Male, 57, Mitral Valve",
            "data": {
                "age": 57,
                "gender": "Men",
                "valve_position": "Mechanical Mitral",
                "current_dose": 3.5,
                "current_inr": 1.7,
                "target_inr_min": 2.5,
                "target_inr_max": 3.5
            },
            "expected_dose": 4.0
        },
        {
            "name": "Patient 3 - Female, 41, Aortic Valve, High INR",
            "data": {
                "age": 41,
                "gender": "Female",
                "valve_position": "Mechanical Aortic",
                "current_dose": 2.44,
                "current_inr": 6.5,
                "target_inr_min": 2.0,
                "target_inr_max": 3.0
            },
            "expected_dose": 6.5
        }
    ]
    
    results = []
    
    for patient in real_patients:
        print(f"\n{'-'*80}")
        print(f"Testing: {patient['name']}")
        print(f"{'-'*80}")
        
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json=patient['data'],
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code == 200:
            result = response.json()
            predicted = result['prediction']['recommended_dose']
            expected = patient['expected_dose']
            error = abs(predicted - expected)
            
            print(f"Current Dose: {patient['data']['current_dose']} mg")
            print(f"Current INR: {patient['data']['current_inr']}")
            print(f"Expected Dose: {expected} mg")
            print(f"Predicted Dose: {predicted} mg")
            print(f"Absolute Error: {error:.2f} mg")
            
            results.append({
                'patient': patient['name'],
                'expected': expected,
                'predicted': predicted,
                'error': error
            })
        else:
            print(f"Error: {response.json()}")
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY OF REAL DATASET TESTS")
    print(f"{'='*80}")
    
    df_results = pd.DataFrame(results)
    print(df_results.to_string(index=False))
    print(f"\nMean Absolute Error: {df_results['error'].mean():.3f} mg")
    print(f"Max Error: {df_results['error'].max():.3f} mg")
    
    return True

# ============================================================================
# RUN ALL TESTS
# ============================================================================

def run_all_tests():
    """Run all test cases"""
    print("\n" + "="*80)
    print(" WARFARIN API COMPREHENSIVE TESTING")
    print("="*80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"API URL: {API_BASE_URL}")
    
    tests = [
        ("Health Check", test_health_check),
        ("Model Info", test_model_info),
        ("Single Prediction (INR Below)", test_single_prediction),
        ("Single Prediction (INR Above)", test_inr_above_range),
        ("Single Prediction (INR In Range)", test_inr_in_range),
        ("Batch Prediction", test_batch_prediction),
        ("Validation Errors", test_validation_errors),
        ("Real Dataset Examples", test_real_dataset)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append({
                'test': test_name,
                'status': 'PASS' if success else 'FAIL'
            })
        except Exception as e:
            print(f"\nTest Failed: {test_name}")
            print(f"Error: {str(e)}")
            results.append({
                'test': test_name,
                'status': 'ERROR'
            })
    
    # Final Summary
    print("\n" + "="*80)
    print(" TEST SUMMARY")
    print("="*80)
    
    df_summary = pd.DataFrame(results)
    print(df_summary.to_string(index=False))
    
    passed = sum(1 for r in results if r['status'] == 'PASS')
    failed = sum(1 for r in results if r['status'] in ['FAIL', 'ERROR'])
    
    print(f"\n{'='*80}")
    print(f"Total Tests: {len(results)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Success Rate: {(passed/len(results)*100):.1f}%")
    print(f"{'='*80}")
    print(f"\nEnd Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ============================================================================
# USAGE EXAMPLES
# ============================================================================

def print_usage_examples():
    """Print usage examples"""
    print("\n" + "="*80)
    print(" API USAGE EXAMPLES")
    print("="*80)
    
    examples = """
# EXAMPLE 1: Python requests
import requests

response = requests.post(
    'http://localhost:5000/predict',
    json={
        "age": 60,
        "gender": "Female",
        "valve_position": "Mechanical Aortic",
        "current_dose": 5.5,
        "current_inr": 2.95,
        "target_inr_min": 2.0,
        "target_inr_max": 3.0
    }
)

print(response.json())

# EXAMPLE 2: cURL
curl -X POST http://localhost:5000/predict \\
  -H "Content-Type: application/json" \\
  -d '{
    "age": 60,
    "gender": "Female",
    "valve_position": "Mechanical Aortic",
    "current_dose": 5.5,
    "current_inr": 2.95,
    "target_inr_min": 2.0,
    "target_inr_max": 3.0
  }'

# EXAMPLE 3: JavaScript fetch
fetch('http://localhost:5000/predict', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    age: 60,
    gender: "Female",
    valve_position: "Mechanical Aortic",
    current_dose: 5.5,
    current_inr: 2.95,
    target_inr_min: 2.0,
    target_inr_max: 3.0
  })
})
.then(response => response.json())
.then(data => console.log(data));
"""
    
    print(examples)

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "examples":
            print_usage_examples()
        else:
            print("Usage: python test_api.py [examples]")
    else:
        run_all_tests()