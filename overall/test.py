import requests
import json
from datetime import datetime

API_URL = "http://localhost:6000"

def print_section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

def print_result(result):
    if not result.get('success', False):
        print(f"\nERROR: {result.get('error', 'Unknown error')}")
        return
    
    print(f"\nPredicted INR Status      : {result['predicted_status']}")
    print(f"Prediction Confidence    : {result['confidence']}")
    print(f"Risk Level               : {result['risk_level']}")
    
    print(f"\nStatus Probability Distribution:")
    for status, prob in result['all_probabilities'].items():
        bar_length = int(prob / 5)
        bar = "#" * bar_length
        print(f"  {status:30s} {prob:6.2f}% {bar}")
    
    print(f"\nTop Contributing Factors:")
    for i, reason in enumerate(result['top_contributing_factors'], 1):
        print(f"\n{i}. Factor      : {reason['factor']}")
        print(f"   Observed Value: {reason['value']}")
        print(f"   Importance    : {reason['importance']}")
        print(f"   Impact        : {reason['impact']}")
    
    print(f"\nRecommended Actions:")
    for rec in result['recommendations']:
        print(f"  - {rec}")

print_section("INR PREDICTION API - AUTOMATED TEST SUITE")
print(f"Test execution started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Target API URL: {API_URL}")

# ---------------------------------------------------------------------
# Test 0: Health Check
# ---------------------------------------------------------------------
print_section("TEST 0: API HEALTH CHECK")

try:
    response = requests.get(f"{API_URL}/health", timeout=5)
    health = response.json()
    print(f"API Status        : {health['status']}")
    print(f"Model Loaded      : {health['model_loaded']}")
    print(f"Model Type        : {health['model_type']}")
    print(f"Number of Features: {health['n_features']}")
    print(f"Number of Classes : {health['n_classes']}")
except requests.exceptions.ConnectionError:
    print("ERROR: Unable to connect to the API.")
    print("Ensure the server is running using: python app.py")
    exit(1)
except Exception as e:
    print(f"Unexpected error occurred: {str(e)}")
    exit(1)

# ---------------------------------------------------------------------
# Test Case 1: Persistently Low INR
# ---------------------------------------------------------------------
print_section("TEST CASE 1: Persistently Low INR")
print("Expected Outcome: Persistently low")

test_case_1 = {
    'Valve Type': 'Mechanical Aortic',
    'Standard Range': '2.0-3.0',
    'Missed Dose Frequency Category': 'nan',
    'What was the reason for missing the dose?': 'nan',
    'Did you take any action after realizing the missed dose?': 'nan',
    'Did you take any extra doses by mistake?': 'No',
    'Did you notice any symptoms': 'Shivering',
    'Do you consume Vitamin K foods or not?': 'Yes',
    'Leafy Greens Consumption Category': 'High (5+ times)',
    'Portion Size Category': 'Large (25g-60g)',
    'Have you made any major dietary changes recently?': 'Yes',
    'Have you started any new antibiotics medicines recommended by a doctor recently?': 'No',
    'Name of the medicine': 'nan',
    'Who recommended it?': 'nan',
    'Are you still taking it?': 'nan'
}

response = requests.post(f"{API_URL}/predict", json=test_case_1)
print_result(response.json())

# ---------------------------------------------------------------------
# Test Case 2: Out of Range Decreased
# ---------------------------------------------------------------------
print_section("TEST CASE 2: Out of Range Decreased")
print("Expected Outcome: Out of range decreased")

test_case_2 = {
    'Valve Type': 'Mechanical Mitral',
    'Standard Range': '2.5-3.5',
    'Missed Dose Frequency Category': 'Rarely (1-2 times)',
    'What was the reason for missing the dose?': 'Busy schedule',
    'Did you take any action after realizing the missed dose?': 'Took extra dose',
    'Did you take any extra doses by mistake?': 'No',
    'Did you notice any symptoms': 'Shivering, Muscle pain, Headache',
    'Do you consume Vitamin K foods or not?': 'Yes',
    'Leafy Greens Consumption Category': 'None',
    'Portion Size Category': 'Small (8g-15g)',
    'Have you made any major dietary changes recently?': 'No',
    'Have you started any new antibiotics medicines recommended by a doctor recently?': 'Yes',
    'Name of the medicine': 'Cetirizine',
    'Who recommended it?': 'Doctor',
    'Are you still taking it?': 'No'
}

response = requests.post(f"{API_URL}/predict", json=test_case_2)
print_result(response.json())

# ---------------------------------------------------------------------
# Test Case 3: Same in Range but Increased
# ---------------------------------------------------------------------
print_section("TEST CASE 3: Same in Range but Increased")
print("Expected Outcome: Same in range but increased")

test_case_3 = {
    'Valve Type': 'Mechanical Mitral',
    'Standard Range': '2.5-3.5',
    'Missed Dose Frequency Category': 'nan',
    'What was the reason for missing the dose?': 'nan',
    'Did you take any action after realizing the missed dose?': 'nan',
    'Did you take any extra doses by mistake?': 'Yes',
    'Did you notice any symptoms': 'Muscle aches',
    'Do you consume Vitamin K foods or not?': 'Yes',
    'Leafy Greens Consumption Category': 'Moderate (3-4 times)',
    'Portion Size Category': 'Large (25g-60g)',
    'Have you made any major dietary changes recently?': 'No',
    'Have you started any new antibiotics medicines recommended by a doctor recently?': 'No',
    'Name of the medicine': 'nan',
    'Who recommended it?': 'nan',
    'Are you still taking it?': 'nan'
}

response = requests.post(f"{API_URL}/predict", json=test_case_3)
print_result(response.json())

# ---------------------------------------------------------------------
# Model Information
# ---------------------------------------------------------------------
print_section("TEST CASE 7: MODEL INFORMATION")

response = requests.get(f"{API_URL}/model-info")
model_info = response.json()

print(f"Model Type            : {model_info['model_type']}")
print(f"Number of Estimators  : {model_info['n_estimators']}")
print(f"Maximum Tree Depth    : {model_info['max_depth']}")
print(f"Number of Features    : {model_info['n_features']}")
print(f"Number of Classes     : {model_info['n_classes']}")

print("\nTarget Classes:")
for i, cls in enumerate(model_info['target_classes'], 1):
    print(f"  {i}. {cls}")

print("\nTop 5 Important Features:")
sorted_features = sorted(
    model_info['feature_importance'],
    key=lambda x: x['Importance'],
    reverse=True
)
for i, feat in enumerate(sorted_features[:5], 1):
    print(f"  {i}. {feat['Feature']} (Importance: {feat['Importance']:.4f})")

# ---------------------------------------------------------------------
# Error Handling Test
# ---------------------------------------------------------------------
print_section("TEST CASE 9: ERROR HANDLING - INCOMPLETE INPUT")

incomplete_data = {
    'Valve Type': 'Mechanical Aortic',
    'Standard Range': '2.0-3.0'
}

response = requests.post(f"{API_URL}/predict", json=incomplete_data)
error_result = response.json()

if not error_result.get('success'):
    print("Input validation working as expected.")
    print(f"Error message       : {error_result.get('error')}")
    print(f"Missing features    : {error_result.get('missing_features', [])[:3]} (showing first 3)")
else:
    print("WARNING: API accepted incomplete input unexpectedly.")

# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------
print_section("TEST EXECUTION SUMMARY")
print(f"All tests completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("\nTest Coverage:")
print("  - Health check")
print("  - Multiple INR prediction scenarios")
print("  - Model metadata validation")
print("  - Input validation and error handling")

print("\nTest suite execution completed successfully.")
print("=" * 80)
