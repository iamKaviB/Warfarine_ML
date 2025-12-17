"""
Comprehensive Test Suite for Heart Rate REST API
Tests with 10 real patient samples from actual dataset
"""

import requests
import json
import sys
import time
import os
from datetime import datetime

# ANSI Colors
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

# Configuration
BASE_URL = os.environ.get('API_URL', 'http://localhost:5002')
NUM_DAYS = 29

def print_header(text):
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'=' * 80}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text.center(80)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'=' * 80}{Colors.ENDC}\n")

def print_success(text):
    print(f"{Colors.OKGREEN}✓ {text}{Colors.ENDC}")

def print_error(text):
    print(f"{Colors.FAIL}✗ {text}{Colors.ENDC}")

def print_info(text):
    print(f"{Colors.OKCYAN}ℹ {text}{Colors.ENDC}")

def print_warning(text):
    print(f"{Colors.WARNING}⚠ {text}{Colors.ENDC}")

def get_all_patient_samples():
    """
    Real patient samples from actual Heart Rate dataset
    10 samples total: okay (rows 0,1,2,5,11,18,19), bad (rows 3,4,8,10,12,25), warning (rows 6,13,14,15,16,17,20,28,30)
    Selected diverse samples representing each category
    """
    
    return {
        # OKAY PATIENTS (3 samples)
        'okay_1': {
            'label': 'okay',
            'description': 'Patient 1 (Row 0) - Stable, minor dip at day 9',
            'data': {
                'val_1': 98, 'val_2': 101, 'val_3': 103, 'val_4': 99, 'val_5': 102,
                'val_6': 107, 'val_7': 104, 'val_8': 101, 'val_9': 30, 'val_10': 106,
                'val_11': 106, 'val_12': 105, 'val_13': 103, 'val_14': 111, 'val_15': 95,
                'val_16': 108, 'val_17': 102, 'val_18': 95, 'val_19': 99, 'val_20': 102,
                'val_21': 102, 'val_22': 106, 'val_23': 107, 'val_24': 117, 'val_25': 108,
                'val_26': 100, 'val_27': 106, 'val_28': 104, 'val_29': 92
            }
        },
        'okay_2': {
            'label': 'okay',
            'description': 'Patient 2 (Row 1) - Very stable 103-132 range',
            'data': {
                'val_1': 120, 'val_2': 121, 'val_3': 105, 'val_4': 109, 'val_5': 103,
                'val_6': 108, 'val_7': 114, 'val_8': 120, 'val_9': 125, 'val_10': 120,
                'val_11': 118, 'val_12': 113, 'val_13': 126, 'val_14': 114, 'val_15': 116,
                'val_16': 113, 'val_17': 108, 'val_18': 115, 'val_19': 121, 'val_20': 104,
                'val_21': 108, 'val_22': 111, 'val_23': 132, 'val_24': 107, 'val_25': 112,
                'val_26': 107, 'val_27': 109, 'val_28': 119, 'val_29': 118
            }
        },
        'okay_3': {
            'label': 'okay',
            'description': 'Patient 19 (Row 18) - Consistent 111-143 range',
            'data': {
                'val_1': 122, 'val_2': 118, 'val_3': 125, 'val_4': 111, 'val_5': 115,
                'val_6': 119, 'val_7': 131, 'val_8': 123, 'val_9': 125, 'val_10': 124,
                'val_11': 120, 'val_12': 117, 'val_13': 128, 'val_14': 129, 'val_15': 131,
                'val_16': 143, 'val_17': 123, 'val_18': 120, 'val_19': 131, 'val_20': 117,
                'val_21': 127, 'val_22': 108, 'val_23': 132, 'val_24': 128, 'val_25': 131,
                'val_26': 111, 'val_27': 125, 'val_28': 111, 'val_29': 117
            }
        },
        
        # BAD PATIENTS (4 samples)
        'bad_1': {
            'label': 'bad',
            'description': 'Patient 4 (Row 3) - Severe escalation 85-196 BPM',
            'data': {
                'val_1': 90, 'val_2': 93, 'val_3': 86, 'val_4': 94, 'val_5': 89,
                'val_6': 94, 'val_7': 85, 'val_8': 97, 'val_9': 111, 'val_10': 93,
                'val_11': 98, 'val_12': 85, 'val_13': 118, 'val_14': 112, 'val_15': 129,
                'val_16': 122, 'val_17': 130, 'val_18': 147, 'val_19': 163, 'val_20': 149,
                'val_21': 170, 'val_22': 172, 'val_23': 167, 'val_24': 178, 'val_25': 196,
                'val_26': 154, 'val_27': 86, 'val_28': 93, 'val_29': 100
            }
        },
        'bad_2': {
            'label': 'bad',
            'description': 'Patient 5 (Row 4) - Drastic crash 117->30 BPM',
            'data': {
                'val_1': 107, 'val_2': 117, 'val_3': 118, 'val_4': 117, 'val_5': 103,
                'val_6': 115, 'val_7': 91, 'val_8': 88, 'val_9': 77, 'val_10': 87,
                'val_11': 79, 'val_12': 75, 'val_13': 81, 'val_14': 68, 'val_15': 76,
                'val_16': 47, 'val_17': 50, 'val_18': 51, 'val_19': 65, 'val_20': 54,
                'val_21': 30, 'val_22': 70, 'val_23': 108, 'val_24': 96, 'val_25': 110,
                'val_26': 116, 'val_27': 96, 'val_28': 112, 'val_29': 109
            }
        },
        'bad_3': {
            'label': 'bad',
            'description': 'Patient 9 (Row 8) - Progressive decline with critical lows',
            'data': {
                'val_1': 108, 'val_2': 111, 'val_3': 104, 'val_4': 96, 'val_5': 89,
                'val_6': 87, 'val_7': 97, 'val_8': 82, 'val_9': 101, 'val_10': 94,
                'val_11': 89, 'val_12': 109, 'val_13': 103, 'val_14': 116, 'val_15': 92,
                'val_16': 107, 'val_17': 107, 'val_18': 81, 'val_19': 105, 'val_20': 81,
                'val_21': 47, 'val_22': 57, 'val_23': 62, 'val_24': 58, 'val_25': 66,
                'val_26': 69, 'val_27': 108, 'val_28': 87, 'val_29': 93
            }
        },
        'bad_4': {
            'label': 'bad',
            'description': 'Patient 13 (Row 12) - Dangerous crash 132->34 BPM',
            'data': {
                'val_1': 117, 'val_2': 119, 'val_3': 113, 'val_4': 88, 'val_5': 107,
                'val_6': 96, 'val_7': 89, 'val_8': 66, 'val_9': 83, 'val_10': 85,
                'val_11': 63, 'val_12': 64, 'val_13': 64, 'val_14': 38, 'val_15': 34,
                'val_16': 86, 'val_17': 126, 'val_18': 121, 'val_19': 117, 'val_20': 128,
                'val_21': 121, 'val_22': 127, 'val_23': 120, 'val_24': 133, 'val_25': 123,
                'val_26': 118, 'val_27': 125, 'val_28': 134, 'val_29': 129
            }
        },
        
        # WARNING PATIENTS (3 samples)
        'warning_1': {
            'label': 'warning',
            'description': 'Patient 7 (Row 6) - High volatility, multiple critical lows',
            'data': {
                'val_1': 87, 'val_2': 78, 'val_3': 88, 'val_4': 94, 'val_5': 93,
                'val_6': 103, 'val_7': 84, 'val_8': 90, 'val_9': 95, 'val_10': 91,
                'val_11': 97, 'val_12': 102, 'val_13': 91, 'val_14': 92, 'val_15': 60,
                'val_16': 87, 'val_17': 77, 'val_18': 73, 'val_19': 73, 'val_20': 49,
                'val_21': 73, 'val_22': 75, 'val_23': 67, 'val_24': 43, 'val_25': 74,
                'val_26': 57, 'val_27': 87, 'val_28': 83, 'val_29': 77
            }
        },
        'warning_2': {
            'label': 'warning',
            'description': 'Patient 16 (Row 15) - Fluctuating 61-175 BPM',
            'data': {
                'val_1': 80, 'val_2': 73, 'val_3': 72, 'val_4': 71, 'val_5': 86,
                'val_6': 67, 'val_7': 63, 'val_8': 81, 'val_9': 85, 'val_10': 61,
                'val_11': 78, 'val_12': 75, 'val_13': 71, 'val_14': 67, 'val_15': 69,
                'val_16': 80, 'val_17': 78, 'val_18': 96, 'val_19': 114, 'val_20': 113,
                'val_21': 103, 'val_22': 118, 'val_23': 138, 'val_24': 146, 'val_25': 145,
                'val_26': 175, 'val_27': 124, 'val_28': 74, 'val_29': 77
            }
        },
        'warning_3': {
            'label': 'warning',
            'description': 'Patient 17 (Row 16) - Escalating trend 83->159 BPM',
            'data': {
                'val_1': 83, 'val_2': 88, 'val_3': 92, 'val_4': 94, 'val_5': 94,
                'val_6': 90, 'val_7': 109, 'val_8': 123, 'val_9': 106, 'val_10': 116,
                'val_11': 132, 'val_12': 125, 'val_13': 136, 'val_14': 133, 'val_15': 145,
                'val_16': 151, 'val_17': 159, 'val_18': 148, 'val_19': 153, 'val_20': 129,
                'val_21': 91, 'val_22': 89, 'val_23': 93, 'val_24': 92, 'val_25': 97,
                'val_26': 90, 'val_27': 89, 'val_28': 84, 'val_29': 96
            }
        }
    }

def test_api_connection():
    """Test 1: Check if API is running"""
    print_header("TEST 1: API Connection")
    
    try:
        response = requests.get(f'{BASE_URL}/', timeout=5)
        if response.status_code == 200:
            print_success("API is running")
            data = response.json()
            print(f"  Version: {data.get('version', 'N/A')}")
            print(f"  Model Loaded: {data.get('model_loaded', False)}")
            return True
        return False
    except requests.exceptions.ConnectionError:
        print_error("Cannot connect to API")
        print_info(f"Make sure API is running at {BASE_URL}")
        return False
    except Exception as e:
        print_error(f"Error: {str(e)}")
        return False

def test_all_patient_predictions():
    """Test 2: Predictions for all 10 real patient samples"""
    print_header("TEST 2: All Patient Predictions (10 Real Samples)")
    
    samples = get_all_patient_samples()
    results = []
    
    for patient_id, patient in samples.items():
        expected_label = patient['label']
        description = patient['description']
        
        print(f"\n{Colors.BOLD}{patient_id.upper()}: {description}{Colors.ENDC}")
        
        try:
            response = requests.post(
                f'{BASE_URL}/api/predict',
                json=patient['data'],
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                predicted = data['prediction']
                confidence = data['confidence']
                stats = data.get('monthly_stats', {})
                
                # Determine result
                match = predicted == expected_label
                status = "✓ CORRECT" if match else f"✗ MISMATCH (expected {expected_label})"
                color = Colors.OKGREEN if match else Colors.WARNING
                
                print(f"{color}  Predicted: {predicted} | Confidence: {confidence:.1%} | {status}{Colors.ENDC}")
                print(f"  Mean HR: {stats.get('mean', 0):.1f} BPM | Risk: {stats.get('risk_level', 'N/A')}")
                
                results.append({
                    'patient': patient_id,
                    'expected': expected_label,
                    'predicted': predicted,
                    'confidence': confidence,
                    'correct': match
                })
            else:
                print_error(f"  Failed with status {response.status_code}")
                results.append({'patient': patient_id, 'correct': False})
                
        except Exception as e:
            print_error(f"  Error: {str(e)}")
            results.append({'patient': patient_id, 'correct': False})
    
    # Summary
    correct = sum(1 for r in results if r.get('correct', False))
    total = len(results)
    accuracy = (correct / total * 100) if total > 0 else 0
    
    print(f"\n{Colors.BOLD}{'='*80}{Colors.ENDC}")
    print(f"{Colors.BOLD}PREDICTION ACCURACY: {correct}/{total} ({accuracy:.1f}%){Colors.ENDC}")
    
    # Breakdown by category
    for label in ['bad', 'okay', 'warning']:
        label_results = [r for r in results if r.get('expected') == label]
        label_correct = sum(1 for r in label_results if r.get('correct', False))
        label_total = len(label_results)
        if label_total > 0:
            print(f"  {label.capitalize()}: {label_correct}/{label_total} correct")
    
    return accuracy >= 70  # Pass if 70%+ accuracy

def test_monthly_analysis_detailed():
    """Test 3: Detailed monthly analysis for each category"""
    print_header("TEST 3: Detailed Monthly Analysis")
    
    samples = get_all_patient_samples()
    test_samples = {
        'bad': 'bad_1',
        'okay': 'okay_1', 
        'warning': 'warning_1'
    }
    
    all_passed = True
    
    for category, patient_id in test_samples.items():
        patient = samples[patient_id]
        print(f"\n{Colors.BOLD}Analyzing {category.upper()} Patient: {patient['description']}{Colors.ENDC}")
        
        try:
            data = patient['data'].copy()
            data['patient_id'] = patient_id
            data['month'] = 'November'
            data['year'] = 2024
            
            response = requests.post(
                f'{BASE_URL}/api/monthly/analyze',
                json=data,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Prediction
                pred = result.get('prediction', {})
                print(f"  Prediction: {pred.get('label')} ({pred.get('confidence', 0):.1%})")
                
                # Stats
                stats = result.get('monthly_summary', {})
                print(f"  Mean: {stats.get('mean', 0):.1f} BPM | Min: {stats.get('min', 0):.0f} | Max: {stats.get('max', 0):.0f}")
                print(f"  Range: {stats.get('range', 0):.0f} BPM | Risk: {stats.get('risk_level', 'N/A')}")
                
                # Weekly breakdown
                if 'weekly_breakdown' in result:
                    print(f"  Weekly Trend:")
                    for week in result['weekly_breakdown']:
                        print(f"    Week {week['week']}: {week['mean']:.1f} BPM (Days {week['days_range']})")
                
                print_success("Analysis completed")
            else:
                print_error(f"Failed with status {response.status_code}")
                all_passed = False
                
        except Exception as e:
            print_error(f"Error: {str(e)}")
            all_passed = False
    
    return all_passed

def test_batch_processing():
    """Test 4: Batch prediction with mixed patients"""
    print_header("TEST 4: Batch Prediction (Mixed Patients)")
    
    samples = get_all_patient_samples()
    
    # Select 5 diverse patients for batch
    batch_patients = [
        samples['bad_1']['data'],
        samples['okay_1']['data'],
        samples['warning_1']['data'],
        samples['bad_2']['data'],
        samples['okay_2']['data']
    ]
    
    try:
        print(f"Submitting batch of {len(batch_patients)} patients...")
        
        response = requests.post(
            f'{BASE_URL}/api/predict/batch',
            json={'patients': batch_patients},
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            print_success(f"Batch completed: {data['success_count']} succeeded, {data['error_count']} failed")
            
            if 'results' in data:
                print(f"\n{Colors.BOLD}Results:{Colors.ENDC}")
                for result in data['results']:
                    pred = result['prediction']
                    conf = result['confidence']
                    mean = result['monthly_stats']['mean']
                    print(f"  Patient {result['patient_index']}: {pred} ({conf:.1%}) - Mean: {mean:.1f} BPM")
            
            return data['success_count'] == len(batch_patients)
        else:
            print_error(f"Failed with status {response.status_code}")
            return False
            
    except Exception as e:
        print_error(f"Error: {str(e)}")
        return False

def test_edge_cases():
    """Test 5: Edge cases and validation"""
    print_header("TEST 5: Edge Cases & Validation")
    
    tests_passed = 0
    total_tests = 3
    
    # Test 1: Incomplete data
    print(f"{Colors.BOLD}Test 5a: Incomplete data (only 10 days){Colors.ENDC}")
    try:
        incomplete_data = {f'val_{i}': 80 for i in range(1, 11)}
        response = requests.post(f'{BASE_URL}/api/validate', json=incomplete_data, timeout=5)
        if response.status_code == 400:
            print_success("Correctly rejected incomplete data")
            tests_passed += 1
        else:
            print_warning("Should have rejected")
    except Exception as e:
        print_error(f"Error: {str(e)}")
    
    # Test 2: Out of range values
    print(f"\n{Colors.BOLD}Test 5b: Out of range values (250 BPM){Colors.ENDC}")
    try:
        samples = get_all_patient_samples()
        invalid_data = samples['okay_1']['data'].copy()
        invalid_data['val_1'] = 250  # Invalid - too high
        response = requests.post(f'{BASE_URL}/api/validate', json=invalid_data, timeout=5)
        if response.status_code == 400:
            print_success("Correctly rejected invalid values")
            tests_passed += 1
        else:
            print_warning("Should have rejected")
    except Exception as e:
        print_error(f"Error: {str(e)}")
    
    # Test 3: Valid edge case (all normal 80 BPM)
    print(f"\n{Colors.BOLD}Test 5c: Edge case - all 80 BPM (normal){Colors.ENDC}")
    try:
        normal_data = {f'val_{i}': 80 for i in range(1, NUM_DAYS + 1)}
        response = requests.post(f'{BASE_URL}/api/validate', json=normal_data, timeout=5)
        if response.status_code == 200:
            print_success("Correctly accepted normal heart rate")
            tests_passed += 1
        else:
            print_warning("Should have accepted")
    except Exception as e:
        print_error(f"Error: {str(e)}")
    
    print(f"\n{Colors.BOLD}Edge case tests: {tests_passed}/{total_tests} passed{Colors.ENDC}")
    return tests_passed >= 2

def run_all_tests():
    """Run comprehensive test suite"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}")
    print("╔════════════════════════════════════════════════════════════════════════════════╗")
    print("║       Heart Rate API - COMPREHENSIVE TEST SUITE (10 Real Patients)            ║")
    print("║                   Bad (4) | Okay (3) | Warning (3)                            ║")
    print("╚════════════════════════════════════════════════════════════════════════════════╝")
    print(f"{Colors.ENDC}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"API: {BASE_URL}\n")
    
    results = {
        'API Connection': test_api_connection(),
        'All Patient Predictions': test_all_patient_predictions(),
        'Monthly Analysis': test_monthly_analysis_detailed(),
        'Batch Processing': test_batch_processing(),
        'Edge Cases': test_edge_cases()
    }
    
    # Summary
    print_header("FINAL TEST SUMMARY")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        if result:
            print_success(f"{test_name}: PASSED")
        else:
            print_error(f"{test_name}: FAILED")
    
    print(f"\n{Colors.BOLD}Overall: {passed}/{total} test suites passed{Colors.ENDC}")
    
    if passed == total:
        print(f"\n{Colors.OKGREEN}{Colors.BOLD}✓✓✓ ALL TESTS PASSED ✓✓✓{Colors.ENDC}")
        print(f"{Colors.OKGREEN}System validated with 10 real patient samples!{Colors.ENDC}")
    else:
        print(f"\n{Colors.WARNING}{Colors.BOLD}⚠ SOME TESTS FAILED ⚠{Colors.ENDC}")
    
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return 0 if passed == total else 1

if __name__ == '__main__':
    exit_code = run_all_tests()
    sys.exit(exit_code)