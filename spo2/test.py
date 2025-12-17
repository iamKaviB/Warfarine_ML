"""
Comprehensive Test Suite for SpO2 REST API
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
BASE_URL = os.environ.get('API_URL', 'http://localhost:5001')
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
    Real patient samples from actual SpO2 dataset
    10 samples total: bad (rows 1,5,9), okay (rows 2,4,8), warning (rows 3,6,7,10)
    """
    
    return {
        # BAD PATIENTS (3 samples)
        'bad_1': {
            'label': 'bad',
            'description': 'Patient 1 - Severe declining trend',
            'data': {
                'val_1': 99, 'val_2': 100, 'val_3': 99, 'val_4': 99, 'val_5': 99,
                'val_6': 99, 'val_7': 100, 'val_8': 99, 'val_9': 99, 'val_10': 100,
                'val_11': 99, 'val_12': 98, 'val_13': 98, 'val_14': 98, 'val_15': 95,
                'val_16': 96, 'val_17': 98, 'val_18': 95, 'val_19': 97, 'val_20': 93,
                'val_21': 97, 'val_22': 92, 'val_23': 91, 'val_24': 90, 'val_25': 92,
                'val_26': 91, 'val_27': 87, 'val_28': 91, 'val_29': 88
            }
        },
        'bad_2': {
            'label': 'bad',
            'description': 'Patient 5 - Starts high, crashes in week 3-4',
            'data': {
                'val_1': 98, 'val_2': 98, 'val_3': 97, 'val_4': 98, 'val_5': 97,
                'val_6': 100, 'val_7': 98, 'val_8': 98, 'val_9': 98, 'val_10': 97,
                'val_11': 98, 'val_12': 95, 'val_13': 98, 'val_14': 96, 'val_15': 93,
                'val_16': 92, 'val_17': 92, 'val_18': 91, 'val_19': 90, 'val_20': 88,
                'val_21': 91, 'val_22': 86, 'val_23': 87, 'val_24': 86, 'val_25': 88,
                'val_26': 85, 'val_27': 83, 'val_28': 82, 'val_29': 90
            }
        },
        'bad_3': {
            'label': 'bad',
            'description': 'Patient 9 - Progressive decline with volatility',
            'data': {
                'val_1': 97, 'val_2': 98, 'val_3': 98, 'val_4': 96, 'val_5': 97,
                'val_6': 96, 'val_7': 97, 'val_8': 98, 'val_9': 95, 'val_10': 96,
                'val_11': 98, 'val_12': 96, 'val_13': 96, 'val_14': 97, 'val_15': 97,
                'val_16': 95, 'val_17': 95, 'val_18': 94, 'val_19': 91, 'val_20': 90,
                'val_21': 94, 'val_22': 95, 'val_23': 96, 'val_24': 92, 'val_25': 95,
                'val_26': 92, 'val_27': 87, 'val_28': 91, 'val_29': 95
            }
        },
        
        # OKAY PATIENTS (3 samples)
        'okay_1': {
            'label': 'okay',
            'description': 'Patient 2 - Consistently healthy 95-97%',
            'data': {
                'val_1': 96, 'val_2': 97, 'val_3': 95, 'val_4': 96, 'val_5': 95,
                'val_6': 96, 'val_7': 97, 'val_8': 96, 'val_9': 95, 'val_10': 95,
                'val_11': 98, 'val_12': 97, 'val_13': 94, 'val_14': 95, 'val_15': 97,
                'val_16': 96, 'val_17': 96, 'val_18': 95, 'val_19': 96, 'val_20': 97,
                'val_21': 96, 'val_22': 95, 'val_23': 97, 'val_24': 97, 'val_25': 96,
                'val_26': 94, 'val_27': 96, 'val_28': 96, 'val_29': 95
            }
        },
        'okay_2': {
            'label': 'okay',
            'description': 'Patient 4 - Stable with minor fluctuations',
            'data': {
                'val_1': 98, 'val_2': 96, 'val_3': 95, 'val_4': 95, 'val_5': 96,
                'val_6': 96, 'val_7': 96, 'val_8': 96, 'val_9': 97, 'val_10': 95,
                'val_11': 97, 'val_12': 96, 'val_13': 96, 'val_14': 95, 'val_15': 96,
                'val_16': 95, 'val_17': 95, 'val_18': 96, 'val_19': 95, 'val_20': 97,
                'val_21': 96, 'val_22': 97, 'val_23': 96, 'val_24': 97, 'val_25': 96,
                'val_26': 95, 'val_27': 96, 'val_28': 97, 'val_29': 96
            }
        },
        'okay_3': {
            'label': 'okay',
            'description': 'Patient 8 - Very stable, excellent health',
            'data': {
                'val_1': 97, 'val_2': 98, 'val_3': 98, 'val_4': 96, 'val_5': 97,
                'val_6': 97, 'val_7': 96, 'val_8': 97, 'val_9': 96, 'val_10': 97,
                'val_11': 98, 'val_12': 97, 'val_13': 97, 'val_14': 97, 'val_15': 97,
                'val_16': 95, 'val_17': 95, 'val_18': 94, 'val_19': 91, 'val_20': 94,
                'val_21': 92, 'val_22': 87, 'val_23': 95, 'val_24': 91, 'val_25': 87,
                'val_26': 95, 'val_27': 92, 'val_28': 95, 'val_29': 97
            }
        },
        
        # WARNING PATIENTS (4 samples)
        'warning_1': {
            'label': 'warning',
            'description': 'Patient 3 - Borderline with mild decline',
            'data': {
                'val_1': 97, 'val_2': 97, 'val_3': 98, 'val_4': 99, 'val_5': 96,
                'val_6': 96, 'val_7': 97, 'val_8': 96, 'val_9': 95, 'val_10': 96,
                'val_11': 94, 'val_12': 95, 'val_13': 96, 'val_14': 94, 'val_15': 97,
                'val_16': 92, 'val_17': 93, 'val_18': 94, 'val_19': 92, 'val_20': 95,
                'val_21': 94, 'val_22': 93, 'val_23': 93, 'val_24': 93, 'val_25': 94,
                'val_26': 96, 'val_27': 97, 'val_28': 99, 'val_29': 98
            }
        },
        'warning_2': {
            'label': 'warning',
            'description': 'Patient 6 - Starts good, dips mid-month, recovers',
            'data': {
                'val_1': 96, 'val_2': 95, 'val_3': 94, 'val_4': 96, 'val_5': 93,
                'val_6': 94, 'val_7': 94, 'val_8': 91, 'val_9': 91, 'val_10': 89,
                'val_11': 93, 'val_12': 98, 'val_13': 97, 'val_14': 97, 'val_15': 97,
                'val_16': 98, 'val_17': 98, 'val_18': 96, 'val_19': 98, 'val_20': 96,
                'val_21': 97, 'val_22': 96, 'val_23': 97, 'val_24': 98, 'val_25': 99,
                'val_26': 96, 'val_27': 97, 'val_28': 98, 'val_29': 95
            }
        },
        'warning_3': {
            'label': 'warning',
            'description': 'Patient 7 - Fluctuating between 93-98%',
            'data': {
                'val_1': 97, 'val_2': 97, 'val_3': 98, 'val_4': 97, 'val_5': 97,
                'val_6': 98, 'val_7': 98, 'val_8': 99, 'val_9': 98, 'val_10': 99,
                'val_11': 97, 'val_12': 98, 'val_13': 97, 'val_14': 97, 'val_15': 97,
                'val_16': 99, 'val_17': 98, 'val_18': 95, 'val_19': 95, 'val_20': 94,
                'val_21': 91, 'val_22': 90, 'val_23': 94, 'val_24': 92, 'val_25': 87,
                'val_26': 88, 'val_27': 87, 'val_28': 91, 'val_29': 98
            }
        },
        'warning_4': {
            'label': 'warning',
            'description': 'Patient 10 - Highly variable, unstable readings',
            'data': {
                'val_1': 96, 'val_2': 96, 'val_3': 97, 'val_4': 97, 'val_5': 97,
                'val_6': 96, 'val_7': 97, 'val_8': 98, 'val_9': 95, 'val_10': 96,
                'val_11': 98, 'val_12': 96, 'val_13': 96, 'val_14': 95, 'val_15': 97,
                'val_16': 95, 'val_17': 94, 'val_18': 97, 'val_19': 93, 'val_20': 94,
                'val_21': 95, 'val_22': 96, 'val_23': 92, 'val_24': 95, 'val_25': 95,
                'val_26': 92, 'val_27': 92, 'val_28': 95, 'val_29': 97
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
                print(f"  Mean SpO2: {stats.get('mean', 0):.1f}% | Risk: {stats.get('risk_level', 'N/A')}")
                
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
                print(f"  Mean: {stats.get('mean', 0):.1f}% | Min: {stats.get('min', 0):.0f}% | Max: {stats.get('max', 0):.0f}%")
                print(f"  Days <90%: {stats.get('days_below_90', 0)} | Risk: {stats.get('risk_level', 'N/A')}")
                
                # Weekly breakdown
                if 'weekly_breakdown' in result:
                    print(f"  Weekly Trend:")
                    for week in result['weekly_breakdown']:
                        print(f"    Week {week['week']}: {week['mean']:.1f}% (Days {week['days_range']})")
                
                # Insights
                if result.get('insights'):
                    print(f"  Top Insight: {result['insights'][0]}")
                
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
                    print(f"  Patient {result['patient_index']}: {pred} ({conf:.1%}) - Mean: {mean:.1f}%")
            
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
        incomplete_data = {f'val_{i}': 95 for i in range(1, 11)}
        response = requests.post(f'{BASE_URL}/api/validate', json=incomplete_data, timeout=5)
        if response.status_code == 400:
            print_success("Correctly rejected incomplete data")
            tests_passed += 1
        else:
            print_warning("Should have rejected")
    except Exception as e:
        print_error(f"Error: {str(e)}")
    
    # Test 2: Out of range values
    print(f"\n{Colors.BOLD}Test 5b: Out of range values{Colors.ENDC}")
    try:
        samples = get_all_patient_samples()
        invalid_data = samples['okay_1']['data'].copy()
        invalid_data['val_1'] = 150  # Invalid
        response = requests.post(f'{BASE_URL}/api/validate', json=invalid_data, timeout=5)
        if response.status_code == 400:
            print_success("Correctly rejected invalid values")
            tests_passed += 1
        else:
            print_warning("Should have rejected")
    except Exception as e:
        print_error(f"Error: {str(e)}")
    
    # Test 3: Valid edge case (all 100%)
    print(f"\n{Colors.BOLD}Test 5c: Edge case - all 100% SpO2{Colors.ENDC}")
    try:
        perfect_data = {f'val_{i}': 100 for i in range(1, NUM_DAYS + 1)}
        response = requests.post(f'{BASE_URL}/api/validate', json=perfect_data, timeout=5)
        if response.status_code == 200:
            print_success("Correctly accepted perfect SpO2")
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
    print("║         SpO2 API - COMPREHENSIVE TEST SUITE (10 Real Patients)                ║")
    print("║                   Bad (3) | Okay (3) | Warning (4)                            ║")
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