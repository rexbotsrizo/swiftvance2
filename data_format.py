# -----------------------------------------------------------
# data_format.py — AI Function Input Test Cases as Python Data
# -----------------------------------------------------------
# Purpose:
# - Define test cases as Python data structures (lists/dicts)
# - Easily import into local_ai.py or ai_logic.py for testing
# - Avoid JSON parsing overhead and allow direct Python usage
#
# Usage:
# - Define a variable (e.g., test_cases) as a list of dictionaries
# - Each dictionary includes:
#     - 'name': unique test case identifier (string)
#     - 'function': target function name (string)
#     - 'input': dict of arguments to pass to the function
#     - 'expected_keys': list of keys expected in function output (optional)
#
# Import and Usage in local_ai.py:
# from data_format import test_cases
#
# for case in test_cases:
#     func_name = case['function']
#     args = case['input']
#     # call function dynamically or via if-else and pass args
#
# Notes:
# - Keep data_format.py synced with your actual AI function signatures
# - This approach enables easier debugging and integration
# -----------------------------------------------------------

test_cases = [
    # Triage AI Tests - Decision Making
    {
        "name": "triage_simple_thanks",
        "function": "triage_message",
        "input": {
            "message": "ok thanks",
            "client_data": {
                "name": "Carlos",
                "case_manager": "Maria",
                "accident_date": "2025-06-01",
                "gender": "male"
            },
            "conversation_history": []
        },
        "expected_keys": ["action", "should_respond", "should_flag", "risk_level", "sentiment", "success"]
    },
    {
        "name": "triage_legal_advice_request",
        "function": "triage_message", 
        "input": {
            "message": "Can I sue the other driver for more money?",
            "client_data": {
                "name": "Ben",
                "case_manager": "Jessica", 
                "accident_date": "2025-05-15",
                "gender": "male"
            },
            "conversation_history": []
        },
        "expected_keys": ["action", "should_respond", "should_flag", "risk_level", "sentiment", "success"]
    },
    {
        "name": "triage_angry_client",
        "function": "triage_message",
        "input": {
            "message": "This is ridiculous! I haven't heard from anyone in weeks and I'm getting bills!",
            "client_data": {
                "name": "Sophia",
                "case_manager": "Kevin",
                "accident_date": "2025-04-20",
                "gender": "female"
            },
            "conversation_history": []
        },
        "expected_keys": ["action", "should_respond", "should_flag", "risk_level", "sentiment", "success"]
    },
    
    # Response Generation Tests
    {
        "name": "generate_empathetic_response",
        "function": "generate_response",
        "input": {
            "message": "I'm still in a lot of pain and feeling frustrated",
            "client_data": {
                "name": "Carlos",
                "case_manager": "Maria",
                "accident_date": "2025-06-01", 
                "gender": "male"
            },
            "conversation_history": [],
            "message_count_this_week": 3
        },
        "expected_keys": ["response", "action_items", "success", "message"]
    },
    {
        "name": "generate_check_in_message",
        "function": "generate_check_in",
        "input": {
            "client_data": {
                "name": "Sophia",
                "case_manager": "Kevin",
                "accident_date": "2025-05-01",
                "gender": "female"
            },
            "days_since_accident": 72,
            "last_sentiment": "neutral"
        },
        "expected_keys": ["message", "success"]
    },
    
    # Sentiment Analysis Tests
    {
        "name": "analyze_positive_sentiment",
        "function": "analyze_sentiment",
        "input": {
            "message": "Thank you so much! Maria has been amazing and everything is going great!"
        },
        "expected_keys": ["sentiment", "confidence", "keywords", "success"]
    },
    {
        "name": "analyze_negative_sentiment", 
        "function": "analyze_sentiment",
        "input": {
            "message": "I'm really unhappy with how slow this is going. Nobody returns my calls."
        },
        "expected_keys": ["sentiment", "confidence", "keywords", "success"]
    },
    
    # Risk Assessment Tests
    {
        "name": "assess_client_risk_high",
        "function": "assess_risk_level",
        "input": {
            "sentiment_history": ["negative", "negative", "negative", "neutral"],
            "response_rate": 0.3,
            "keywords_mentioned": ["slow", "unhappy", "frustrated", "leaving"],
            "days_since_last_response": 14
        },
        "expected_keys": ["risk_level", "reasoning", "recommendations", "success"]
    },
    {
        "name": "assess_client_risk_low",
        "function": "assess_risk_level", 
        "input": {
            "sentiment_history": ["positive", "positive", "neutral", "positive"],
            "response_rate": 0.9,
            "keywords_mentioned": ["great", "helpful", "thank you"],
            "days_since_last_response": 2
        },
        "expected_keys": ["risk_level", "reasoning", "recommendations", "success"]
    },
    
    # Client Insight Generation Tests
    {
        "name": "generate_client_insight",
        "function": "generate_client_insight",
        "input": {
            "client_data": {
                "name": "Ben",
                "case_manager": "Jessica",
                "accident_date": "2025-04-15",
                "gender": "male"
            },
            "conversation_history": [
                {"sender": "ai", "message": "Hi Ben, checking in from your team", "timestamp": "2025-07-01"},
                {"sender": "client", "message": "Hey thanks", "timestamp": "2025-07-01"},
                {"sender": "client", "message": "Actually I have a question about costs", "timestamp": "2025-07-01"}
            ],
            "sentiment_trend": ["neutral", "positive", "neutral"],
            "current_risk_level": "medium"
        },
        "expected_keys": ["summary", "key_concerns", "sentiment_trend", "risk_factors", "recommendations", "success"]
    }
]
