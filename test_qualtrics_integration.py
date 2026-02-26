#!/usr/bin/env python
"""Quick test script to demonstrate Qualtrics integration features"""

from urllib.parse import urlparse, urlunparse, parse_qs, urlencode, unquote

def test_url_parameter_handling():
    """Test URL parameter parsing and validation"""
    print("=" * 70)
    print("QUALTRICS INTEGRATION - URL PARAMETER TEST")
    print("=" * 70)
    
    # Simulate different URL scenarios
    test_cases = [
        {
            "name": "Full Qualtrics URL",
            "params": {
                "pid": "P123",
                "cond": "experimental",
                "return": "https%3A%2F%2Fsurvey.qualtrics.com%2Fjfe%2Fform%2FSV_ABC123"
            }
        },
        {
            "name": "Prolific Integration",
            "params": {
                "PROLIFIC_PID": "5f1234567890abc",
                "pid": "5f1234567890abc",
                "cond": "control",
                "return": "https%3A%2F%2Fapp.prolific.com%2Fsubmissions%2Fcomplete%3Fcc%3DXXXXXX"
            }
        },
        {
            "name": "Invalid return URL (should fail)",
            "params": {
                "pid": "TEST",
                "return": "https%3A%2F%2Fmalicious-site.com%2Fsurvey"
            }
        },
        {
            "name": "No return URL (local testing)",
            "params": {
                "pid": "LOCAL_TEST",
                "cond": "test"
            }
        }
    ]
    
    for test in test_cases:
        print(f"\n{test['name']}")
        print("-" * 70)
        
        # Parse parameters
        params = test['params']
        pid = params.get('pid', '')
        cond = params.get('cond', '')
        return_raw = params.get('return', '')
        prolific_pid = params.get('PROLIFIC_PID', '')
        
        print(f"  Participant ID: {pid}")
        print(f"  Condition: {cond}")
        print(f"  Prolific PID: {prolific_pid or 'None'}")
        
        if return_raw:
            # Decode and validate return URL
            decoded = unquote(return_raw)
            print(f"  Return URL (encoded): {return_raw}")
            print(f"  Return URL (decoded): {decoded}")
            
            # Check if it's a valid Qualtrics URL
            try:
                if not decoded.startswith(("http://", "https://")):
                    decoded = "https://" + decoded
                p = urlparse(decoded)
                is_valid = (p.scheme in ("http", "https")) and ("qualtrics.com" in p.netloc or "prolific.com" in p.netloc)
                print(f"  Validation: {'✅ VALID' if is_valid else '❌ INVALID (not Qualtrics/Prolific domain)'}")
                
                if is_valid:
                    # Build final return URL with done=1
                    q = parse_qs(p.query, keep_blank_values=True)
                    q["done"] = "1"
                    if pid and "pid" not in q:
                        q["pid"] = pid
                    if prolific_pid and "PROLIFIC_PID" not in q:
                        q["PROLIFIC_PID"] = prolific_pid
                    
                    final_url = urlunparse(p._replace(query=urlencode(q, doseq=True)))
                    print(f"  Final redirect URL: {final_url}")
            except Exception as e:
                print(f"  Validation: ❌ ERROR - {e}")
        else:
            print(f"  Return URL: None (local testing mode)")
            print(f"  Behavior: Show 'Start New Session' button instead of redirect")


def test_feedback_collection():
    """Demonstrate feedback data structure"""
    print("\n" + "=" * 70)
    print("FEEDBACK DATA STRUCTURE")
    print("=" * 70)
    
    # Example query result with feedback
    example_result = {
        'session_id': 'abc123',
        'query_index': 0,
        'query_text': 'I want to transfer money',
        'true_intent': 'transfer',
        'predicted_intent': 'transfer',
        'confidence': 0.87,
        'num_clarification_turns': 1,
        'is_correct': True,
        'interaction_time_seconds': 23.5,
        'conversation_transcript': 'User: I want to transfer money\nAssistant: I understand you want to do a transfer. To which account?',
        'timestamp': '2026-02-23T10:30:00',
        'participant_id': 'P123',
        'condition': 'experimental',
        'prolific_pid': '5f1234567890abc',
        'feedback_clarity': 5,
        'feedback_confidence': 4,
        'feedback_comment': 'Very clear explanation',
        'feedback_submitted': True,
        'llm_predicted_intent': 'transfer',
        'llm_num_interactions': 2,
        'llm_confidence': 0.85,
        'llm_was_correct': True
    }
    
    print("\nPer-Query Feedback Fields:")
    print("-" * 70)
    for key in ['feedback_clarity', 'feedback_confidence', 'feedback_comment', 'feedback_submitted']:
        print(f"  {key}: {example_result[key]}")
    
    print("\nParticipant Tracking Fields:")
    print("-" * 70)
    for key in ['session_id', 'participant_id', 'condition', 'prolific_pid']:
        print(f"  {key}: {example_result[key]}")
    
    print("\nPerformance Metrics:")
    print("-" * 70)
    for key in ['is_correct', 'num_clarification_turns', 'interaction_time_seconds', 'confidence']:
        print(f"  {key}: {example_result[key]}")
    
    # Example final feedback
    example_final = {
        "overall_rating": 4,
        "trust": 4,
        "ease_of_use": 5,
        "would_recommend": "Probably",
        "additional_comments": "Great system, very intuitive!",
        "session_id": "abc123",
        "participant_id": "P123",
        "condition": "experimental",
        "prolific_pid": "5f1234567890abc",
        "timestamp": "2026-02-23T10:45:00",
        "num_queries_completed": 100,
        "accuracy": 0.92,
        "avg_clarifications": 1.3,
        "avg_time_seconds": 25.7
    }
    
    print("\n" + "=" * 70)
    print("FINAL SURVEY DATA")
    print("=" * 70)
    for key, value in example_final.items():
        print(f"  {key}: {value}")


def generate_example_urls():
    """Generate example URLs for different study setups"""
    print("\n" + "=" * 70)
    print("EXAMPLE STUDY URLS")
    print("=" * 70)
    
    base_url = "https://your-app.streamlit.app/"
    
    scenarios = [
        {
            "name": "Prolific Study",
            "params": {
                "pid": "{{%PROLIFIC_PID%}}",
                "cond": "treatment_A",
                "return": "https://app.prolific.com/submissions/complete?cc=COMPLETION_CODE"
            },
            "where": "Use this in Prolific study link"
        },
        {
            "name": "Qualtrics A/B Test",
            "params": {
                "pid": "${e://Field/ResponseID}",
                "cond": "${e://Field/condition}",
                "return": "${e://Field/Q_URL}"
            },
            "where": "Use this in Qualtrics Web Service"
        },
        {
            "name": "Local Testing",
            "params": {
                "pid": "TEST_USER_001",
                "cond": "test"
            },
            "where": "Use this for local development testing"
        }
    ]
    
    for scenario in scenarios:
        print(f"\n{scenario['name']}")
        print(f"Context: {scenario['where']}")
        print("-" * 70)
        
        # Build URL
        param_strs = []
        for key, value in scenario['params'].items():
            if key == 'return':
                # URL encode return parameter
                from urllib.parse import quote
                encoded_value = quote(value, safe='')
                param_strs.append(f"{key}={encoded_value}")
            else:
                param_strs.append(f"{key}={value}")
        
        full_url = base_url + "?" + "&".join(param_strs)
        print(full_url)


if __name__ == "__main__":
    test_url_parameter_handling()
    test_feedback_collection()
    generate_example_urls()
    
    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Deploy app to Streamlit Cloud")
    print("2. Test with example URLs above")
    print("3. Set up Qualtrics survey with Web Service")
    print("4. Run pilot study with 3-5 participants")
    print("5. Check outputs/user_study/feedback/ for data")
