# test.py
# Comprehensive test suite for Warfarin Complete Behavior Analysis API - Version 7

import requests
import json
from datetime import datetime
import sys

BASE_URL = "http://localhost:8090"

def print_section(title):
    """Print section header"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)

def print_response(response, show_full=False, max_items=5):
    """Pretty print response"""
    print(f"Status Code: {response.status_code}")
    try:
        data = response.json()
        
        # For large responses, show summary
        if not show_full and isinstance(data, dict):
            print(f"Response Summary:")
            for key, value in data.items():
                if isinstance(value, list):
                    print(f"  {key}: [{len(value)} items]")
                    if len(value) > 0 and len(value) <= max_items:
                        for i, item in enumerate(value[:max_items], 1):
                            if isinstance(item, dict):
                                print(f"    {i}. {list(item.keys())[:3]}...")
                elif isinstance(value, dict):
                    print(f"  {key}: {{{len(value)} keys}}")
                else:
                    print(f"  {key}: {value}")
        else:
            print(f"Response:\n{json.dumps(data, indent=2)}")
    except:
        print(f"Response: {response.text[:500]}")

def test_health_check():
    """Test 1: API health check"""
    print_section("TEST 1: Health Check & API Information")
    
    try:
        response = requests.get(f"{BASE_URL}/")
        print_response(response, show_full=True)
        
        if response.status_code == 200:
            data = response.json()
            print(f"\n✓ API Status: {data['status']}")
            print(f"✓ Service: {data['service']}")
            print(f"✓ Total Behaviors: {data['total_behaviors_analyzed']}")
            
            if data['total_behaviors_analyzed'] == 0:
                print("\n⚠ WARNING: No behaviors analyzed yet!")
                print("Please run: python warfarin_complete_behavior_analysis.py")
                return False
        
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_detailed_health():
    """Test 2: Detailed health check"""
    print_section("TEST 2: Detailed Health Check")
    
    try:
        response = requests.get(f"{BASE_URL}/health")
        print_response(response, show_full=True)
        
        if response.status_code == 200:
            data = response.json()
            print(f"\n📊 Data Status:")
            for key, value in data['data_loaded'].items():
                status = "✓" if value else "✗"
                print(f"  {status} {key}: {value}")
            
            print(f"\n📁 Files Status:")
            for file, exists in data['files_status'].items():
                status = "✓" if exists else "✗"
                print(f"  {status} {file}")
            
            print(f"\n📈 Records Count:")
            for key, count in data['records_count'].items():
                print(f"  • {key}: {count}")
        
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_statistics():
    """Test 3: Overall statistics"""
    print_section("TEST 3: Overall Analysis Statistics")
    
    try:
        response = requests.get(f"{BASE_URL}/statistics")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Response:\n{json.dumps(data, indent=2)}")
            
            print(f"\n📊 STATISTICS SUMMARY:")
            print("-"*80)
            print(f"Total behaviors analyzed: {data['total_behaviors_analyzed']}")
            print(f"Behavior categories: {data['behavior_categories']}")
            print(f"\nBehavior types distribution:")
            for btype, count in data.get('behavior_types', {}).items():
                print(f"  • {btype}: {count}")
            
            print(f"\nSignificant behaviors:")
            sig = data['significant_behaviors']
            print(f"  • p < 0.001: {sig['p_less_than_0.001']} (highly significant)")
            print(f"  • p < 0.01:  {sig['p_less_than_0.01']} (very significant)")
            print(f"  • p < 0.05:  {sig['p_less_than_0.05']} (significant)")
            
            print(f"\nINR Change Rate:")
            inr = data['inr_change_rate_range']
            print(f"  • Min: {inr['min']:.2f}%")
            print(f"  • Max: {inr['max']:.2f}%")
            print(f"  • Mean: {inr['mean']:.2f}%")
            
            print(f"\nCategories analyzed:")
            for cat in data['category_list'][:10]:
                print(f"  • {cat}")
            if len(data['category_list']) > 10:
                print(f"  ... and {len(data['category_list']) - 10} more")
        else:
            print_response(response)
        
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_top_behaviors():
    """Test 4: Get top impactful behaviors"""
    print_section("TEST 4: Top Most Impactful Behaviors")
    
    try:
        # Get top 10
        print("Getting top 10 behaviors...")
        response = requests.get(f"{BASE_URL}/top_behaviors?n=10")
        
        if response.status_code == 200:
            data = response.json()
            
            print(f"\n🏆 TOP 10 BEHAVIORS MOST RELATED TO INR CHANGES:")
            print("="*80)
            
            for i, behavior in enumerate(data['behaviors'][:10], 1):
                sig_marker = "***" if behavior['p_value'] < 0.001 else "**" if behavior['p_value'] < 0.01 else "*" if behavior['p_value'] < 0.05 else ""
                
                print(f"\n{i}. {behavior['behavior_category']}: {behavior['behavior_value']}")
                print(f"   Type: {behavior['behavior_type']}")
                print(f"   Impact Score: {behavior['impact_score']:.4f} {sig_marker}")
                print(f"   INR Change Rate: {behavior['inr_change_rate']:.2f}%")
                print(f"   Significance: {behavior['significance']}")
                print(f"   Cases: {behavior['total_cases']}")
            
            if 'summary' in data and 'key_findings' in data['summary']:
                print(f"\n📋 Key Findings:")
                for finding in data['summary']['key_findings']:
                    print(f"  • {finding}")
        else:
            print_response(response)
        
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_behavior_categories():
    """Test 5: Get behavior categories"""
    print_section("TEST 5: Behavior Categories Overview")
    
    try:
        response = requests.get(f"{BASE_URL}/behavior_categories")
        
        if response.status_code == 200:
            data = response.json()
            
            print(f"\n📋 BEHAVIOR CATEGORIES ({data['total_categories']} total):")
            print("="*80)
            
            for i, cat in enumerate(data['categories'][:15], 1):
                print(f"\n{i}. {cat['category']}")
                print(f"   Number of values: {cat['num_values']}")
                print(f"   Max impact: {cat['max_impact']:.4f}")
                print(f"   Avg INR change rate: {cat['avg_inr_change_rate']:.2f}%")
                print(f"   Best p-value: {cat['min_p_value']:.4f}")
            
            if len(data['categories']) > 15:
                print(f"\n... and {len(data['categories']) - 15} more categories")
        else:
            print_response(response)
        
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_search_behavior():
    """Test 6: Search behaviors"""
    print_section("TEST 6: Search for Specific Behaviors")
    
    search_terms = [
        "vitamin",
        "dose",
        "medication",
        "warfarin"
    ]
    
    success = True
    
    for term in search_terms:
        print(f"\n🔍 Searching for: '{term}'")
        print("-"*80)
        
        try:
            response = requests.post(
                f"{BASE_URL}/search_behavior",
                json={"keyword": term, "search_in": ["category", "value"]},
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"Found {data['total_matches']} matches")
                
                if data['total_matches'] > 0:
                    print("\nTop 5 matches:")
                    for i, match in enumerate(data['matches'][:5], 1):
                        print(f"\n{i}. {match['behavior_category']}: {match['behavior_value']}")
                        print(f"   Impact: {match['impact_score']:.4f}")
                        print(f"   INR Change Rate: {match['inr_change_rate']:.2f}%")
                        print(f"   Significance: {match['significance']}")
                else:
                    print("No matches found")
            else:
                print(f"Status Code: {response.status_code}")
                print_response(response)
                success = False
                
        except Exception as e:
            print(f"❌ Error: {e}")
            success = False
    
    return success

def test_analyze_specific():
    """Test 7: Analyze specific behaviors"""
    print_section("TEST 7: Analyze Specific Behaviors")
    
    test_cases = [
        {
            "name": "Check Vitamin K and Doses",
            "behaviors": [
                {"category": "Vitamin", "value": "Yes"},
                {"category": "Missed", "value": "Yes"},
                {"category": "Over", "value": "Yes"}
            ]
        },
        {
            "name": "Check Medications",
            "behaviors": [
                {"category": "Medication", "value": "Verapamil"},
                {"category": "Medication", "value": "Panadol"}
            ]
        },
        {
            "name": "Check Warfarin Usage",
            "behaviors": [
                {"category": "taking", "value": "Yes"}
            ]
        }
    ]
    
    success = True
    
    for test_case in test_cases:
        print(f"\n📝 Test: {test_case['name']}")
        print("-"*80)
        print(f"Analyzing: {json.dumps(test_case['behaviors'], indent=2)}")
        
        try:
            response = requests.post(
                f"{BASE_URL}/analyze_specific",
                json={"behaviors": test_case['behaviors']},
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"\n✓ Analyzed {data['total_analyzed']} behaviors")
                
                for result in data['results']:
                    if 'interpretation' in result:
                        print(f"\n  ✓ {result['behavior_category']}: {result['behavior_value']}")
                        print(f"     {result['interpretation']}")
                        print(f"     Risk Level: {result.get('risk_level', 'N/A')}")
                        print(f"     Impact Score: {result.get('impact_score', 0):.4f}")
                    elif result.get('status') == 'not found':
                        print(f"\n  ✗ Not found: {result.get('category', 'Unknown')} - {result.get('value', 'Unknown')}")
                        print(f"     {result.get('message', '')}")
                    else:
                        print(f"\n  ? {result.get('status', 'unknown')}: {result}")
            else:
                print_response(response)
                success = False
                
        except Exception as e:
            print(f"❌ Error: {e}")
            success = False
    
    return success

def test_compare_behaviors():
    """Test 8: Compare behaviors"""
    print_section("TEST 8: Compare Multiple Behaviors")
    
    comparison_sets = [
        {
            "name": "Compare All Medications",
            "behaviors": [
                {"category": "Medication", "value": "Verapamil"},
                {"category": "Medication", "value": "Panadol"},
                {"category": "Medication", "value": "Metronidazole"},
                {"category": "Medication", "value": "Amoxicillin"}
            ]
        },
        {
            "name": "Compare Risk Factors",
            "behaviors": [
                {"category": "Vitamin", "value": "Yes"},
                {"category": "Missed", "value": "Yes"},
                {"category": "Over", "value": "Yes"}
            ]
        }
    ]
    
    success = True
    
    for comp_set in comparison_sets:
        print(f"\n⚖️  {comp_set['name']}")
        print("-"*80)
        print(f"Comparing: {json.dumps(comp_set['behaviors'], indent=2)}")
        
        try:
            response = requests.post(
                f"{BASE_URL}/compare_behaviors",
                json={"behaviors": comp_set['behaviors']},
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 200:
                data = response.json()
                
                if 'summary' in data:
                    print(f"\n{data['summary']}")
                
                print(f"\nComparison Results (sorted by impact):")
                print("="*80)
                
                for i, comp in enumerate(data['comparisons'], 1):
                    sig_marker = "***" if comp['p_value'] < 0.001 else "**" if comp['p_value'] < 0.01 else "*" if comp['p_value'] < 0.05 else ""
                    
                    print(f"\n{i}. {comp['category']}: {comp['value']}")
                    print(f"   Impact Score: {comp['impact_score']:.4f} {sig_marker}")
                    print(f"   INR Change Rate: {comp['inr_change_rate']:.2f}%")
                    print(f"   Significance: {comp['significance']}")
                    print(f"   Cases: {comp['total_cases']}")
                
                if 'not_found' in data:
                    print(f"\n⚠ Not found: {data['not_found']}")
            else:
                print_response(response)
                success = False
                
        except Exception as e:
            print(f"❌ Error: {e}")
            success = False
    
    return success

def test_ml_features():
    """Test 9: ML feature importance"""
    print_section("TEST 9: Machine Learning Feature Importance")
    
    try:
        response = requests.get(f"{BASE_URL}/ml_features?n=15")
        
        if response.status_code == 200:
            data = response.json()
            
            print(f"\n🤖 TOP 15 ML FEATURES:")
            print("="*80)
            
            for i, feature in enumerate(data['features'], 1):
                print(f"\n{i}. {feature['feature']}")
                print(f"   Importance: {feature['importance']:.4f}")
                if 'description' in feature:
                    print(f"   Description: {feature['description']}")
        else:
            print_response(response)
        
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_filters():
    """Test 10: Filtered queries"""
    print_section("TEST 10: Query Behaviors with Filters")
    
    filters = [
        {"name": "Significant only (p<0.05)", "url": "/all_behaviors?significant_only=true"},
        {"name": "At least 5 cases", "url": "/all_behaviors?min_cases=5"},
        {"name": "High impact (>1.0)", "url": "/all_behaviors?min_impact=1.0"},
        {"name": "Binary behaviors only", "url": "/all_behaviors?type=Binary"}
    ]
    
    success = True
    
    for filter_set in filters:
        print(f"\n🔧 Filter: {filter_set['name']}")
        print("-"*80)
        
        try:
            response = requests.get(f"{BASE_URL}{filter_set['url']}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"Total behaviors matching filter: {data['total_behaviors']}")
                
                if data['total_behaviors'] > 0:
                    print(f"\nTop 3 matching behaviors:")
                    for i, behavior in enumerate(data['behaviors'][:3], 1):
                        print(f"{i}. {behavior['behavior_category']}: {behavior['behavior_value']}")
                        print(f"   Impact: {behavior['impact_score']:.4f}, INR Change: {behavior['inr_change_rate']:.2f}%")
            else:
                print_response(response)
                success = False
                
        except Exception as e:
            print(f"❌ Error: {e}")
            success = False
    
    return success

def run_all_tests():
    """Run all tests"""
    print("\n" + "="*80)
    print("  WARFARIN BEHAVIOR ANALYSIS - COMPREHENSIVE TEST SUITE V7")
    print("="*80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nTesting ALL behaviors: medications, vitamin K, missed dose,")
    print("overdose, and all other questions in the dataset")
    
    tests = {
        '1. Health Check': test_health_check,
        '2. Detailed Health': test_detailed_health,
        '3. Statistics': test_statistics,
        '4. Top Behaviors': test_top_behaviors,
        '5. Categories': test_behavior_categories,
        '6. Search': test_search_behavior,
        '7. Analyze Specific': test_analyze_specific,
        '8. Compare': test_compare_behaviors,
        '9. ML Features': test_ml_features,
        '10. Filters': test_filters
    }
    
    results = {}
    for test_name, test_func in tests.items():
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n❌ Test '{test_name}' crashed: {e}")
            results[test_name] = False
    
    # Summary
    print_section("TEST SUMMARY")
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    total = len(results)
    passed = sum(results.values())
    
    print(f"\n{'='*80}")
    print(f"Total: {passed}/{total} tests passed ({(passed/total*100):.1f}%)")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    return passed == total

if __name__ == "__main__":
    try:
        print("Checking if API is running...")
        response = requests.get(f"{BASE_URL}/", timeout=2)
        
        if response.status_code == 200:
            print("✅ API is running\n")
            success = run_all_tests()
            sys.exit(0 if success else 1)
        else:
            print("⚠️  API returned unexpected status")
            sys.exit(1)
            
    except requests.exceptions.ConnectionError:
        print("\n" + "="*80)
        print("❌ ERROR: Cannot connect to API")
        print("="*80)
        print(f"\nThe API is not running on {BASE_URL}")
        print("\n📋 SETUP INSTRUCTIONS:")
        print("-"*80)
        print("\n1️⃣  Run the complete behavior analysis:")
        print("   python warfarin_complete_behavior_analysis.py")
        print("\n   This will analyze ALL behaviors in your dataset:")
        print("   • Medications (Verapamil, Panadol, Metronidazole, Amoxicillin)")
        print("   • Vitamin K consumption")
        print("   • Missed doses")
        print("   • Overdoses")
        print("   • Warfarin usage")
        print("   • ALL other columns/questions")
        print("\n2️⃣  Start the Flask API:")
        print("   python app.py")
        print("\n3️⃣  Run this test script:")
        print("   python test.py")
        print("\n" + "="*80)
        sys.exit(1)