# test2.py
# Test multiple real cases against /impact-behavior endpoint

import requests
import json

URL = "http://127.0.0.1:8090/impact-behavior"
HEADERS = {"Content-Type": "application/json"}

# ---------------------------------------------------
# TEST CASES (DERIVED FROM YOUR DATASET)
# ---------------------------------------------------
test_cases = [
    {
        "case_id": "CASE_01_PERSISTENTLY_LOW",
        "expected_status": "Persistently low",
        "payload": {
            "inr_status": 1,
            "behaviors": {
                "Did you miss any warfarin doses this week?": "No",
                "Do you consume Vitamin K foods or not?": "Yes",
                "Leafy Greens Consumption Category": "High (5+ times)",
                "Portion Size Category": "Large (25g-60g)",
                "Have you started any new antibiotics medicines recommended by a doctor recently?": "No"
            }
        }
    },

    {
        "case_id": "CASE_02_OUT_OF_RANGE_DECREASED",
        "expected_status": "Out of range decreased",
        "payload": {
            "inr_status": 1,
            "behaviors": {
                "Did you miss any warfarin doses this week?": "Yes",
                "What was the reason for missing the dose?": "Busy schedule",
                "Missed Dose Frequency Category": "Rarely (1-2 times)",
                "Did you take any extra doses by mistake?": "No",
                "Have you started any new antibiotics medicines recommended by a doctor recently?": "Yes"
            }
        }
    },

    {
        "case_id": "CASE_03_OUT_OF_RANGE_INCREASED",
        "expected_status": "Out of range increased",
        "payload": {
            "inr_status": 1,
            "behaviors": {
                "Did you miss any warfarin doses this week?": "Yes",
                "Missed Dose Frequency Category": "Frequently (>5 times)",
                "Were these doses missed consecutively or on different days?": "Consecutively",
                "Did you take any action after realizing the missed dose?": "Took extra dose",
                "Do you consume Vitamin K foods or not?": "Yes",
                "Leafy Greens Consumption Category": "Low (1-2 times)"
            }
        }
    },

    {
        "case_id": "CASE_04_PERSISTENTLY_HIGH",
        "expected_status": "Persistently high",
        "payload": {
            "inr_status": 1,
            "behaviors": {
                "Did you miss any warfarin doses this week?": "No",
                "Did you take any extra doses by mistake?": "Yes",
                "Do you consume Vitamin K foods or not?": "No",
                "Have you made any major dietary changes recently?": "No",
                "Have you started any new antibiotics medicines recommended by a doctor recently?": "No"
            }
        }
    },

    {
        "case_id": "CASE_05_IMPROVED_TO_RANGE",
        "expected_status": "Improved to range",
        "payload": {
            "inr_status": 0,
            "behaviors": {
                "Did you miss any warfarin doses this week?": "No",
                "Did you take any extra doses by mistake?": "No",
                "Do you consume Vitamin K foods or not?": "Yes",
                "Leafy Greens Consumption Category": "Moderate (3-4 times)",
                "Have you started any new antibiotics medicines recommended by a doctor recently?": "No"
            }
        }
    }
]

# ---------------------------------------------------
# RUN TESTS
# ---------------------------------------------------
for case in test_cases:
    print("\n" + "=" * 80)
    print(f"{case['case_id']}")
    print(f"Expected Clinical Status: {case['expected_status']}")

    try:
        response = requests.post(
            URL,
            data=json.dumps(case["payload"]),
            headers=HEADERS,
            timeout=10
        )

        print("HTTP Status:", response.status_code)

        if response.status_code == 200:
            result = response.json()

            print("\nReturned INR Status:", result.get("inr_status"))
            print("Model Probability (Out of Range):",
                  result.get("model_probability_out_of_range"))

            print("\nTop Impact Behaviors:")
            for b in result.get("top_impact_behaviors", []):
                print(f"  • {b['behavior']} → impact={b['impact_score']}")

        else:
            print("Error:", response.text)

    except Exception as e:
        print("Exception:", str(e))
