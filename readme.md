# AI Logic Testing Repository

## Overview

This repository provides a **clean, modular structure** for developing, testing, and integrating AI-related functions in Python. The design follows a **test-driven development** approach where AI logic is written as pure functions, tested locally, and then integrated into production systems.

## 🎯 Key Benefits

- **Separation of Concerns**: AI logic is isolated from web frameworks and external dependencies
- **Test-Driven Development**: Write tests first, then implement AI functions
- **Reusable Components**: Pure functions that can be used across different projects
- **Easy Integration**: Seamless transition from local testing to production deployment

---

## 📁 Project Structure

```
AI Structure/
├── readme.md           # This comprehensive guide
├── data_format.py      # Test case definitions
├── local_ai.py         # Local development and testing
├── ai_logic.py         # Production-ready AI functions
└── requirements.txt    # (recommended) Package dependencies
```

### File Descriptions

#### `data_format.py` - Test Case Definitions
**Purpose**: Define test cases as Python data structures for consistent testing.

**Contains**:
- `test_cases` list with input/output specifications
- Each test case includes function name, input arguments, and expected output keys
- No actual test execution - just data definitions

**Example Structure**:
```python
test_cases = [
    {
        "name": "summary_test_1",
        "function": "generate_summary",
        "input": {"text": "Long text to summarize..."},
        "expected_keys": ["summary", "word_count"]
    }
]
```

#### `local_ai.py` - Development & Testing Environment
**Purpose**: Local development workspace for writing and testing AI functions.

**Features**:
- Import test cases from `data_format.py`
- Run tests directly via `python local_ai.py`
- Debug and iterate on AI logic
- No external dependencies (Django, Flask, etc.)

**Workflow**:
1. Write AI function
2. Import test cases
3. Run function with test data
4. Debug and refine
5. Repeat until satisfied

#### `ai_logic.py` - Production-Ready Functions
**Purpose**: Clean, tested AI functions ready for production integration.

**Characteristics**:
- Pure functions (no side effects)
- Accept data as function arguments only
- Return structured responses with success/error handling
- Can be imported into Django views, APIs, or other services

---

## 🚀 Getting Started

### Step 1: Define Your Test Cases

Edit `data_format.py` to define what your AI functions should do:

```python
test_cases = [
    {
        "name": "text_classification_basic",
        "function": "classify_text",
        "input": {
            "text": "I love this product! It's amazing!",
            "categories": ["positive", "negative", "neutral"]
        },
        "expected_keys": ["category", "confidence", "success"]
    },
    {
        "name": "text_summarization_short",
        "function": "summarize_text",
        "input": {
            "text": "Very long article text here...",
            "max_length": 100
        },
        "expected_keys": ["summary", "original_length", "success"]
    }
]
```

### Step 2: Develop in `local_ai.py`

Write and test your AI functions locally:

```python
# Import your test cases
from data_format import test_cases

def classify_text(text, categories):
    """Example AI function for text classification"""
    # Your AI logic here
    return {
        "category": "positive",
        "confidence": 0.95,
        "success": True,
        "message": "Classification completed successfully"
    }

def summarize_text(text, max_length):
    """Example AI function for text summarization"""
    # Your AI logic here
    return {
        "summary": "Shortened version of the text...",
        "original_length": len(text),
        "success": True,
        "message": "Summarization completed successfully"
    }

# Test runner
if __name__ == "__main__":
    for case in test_cases:
        func_name = case['function']
        args = case['input']
        expected_keys = case.get('expected_keys', [])
        
        print(f"\n🧪 Testing: {case['name']}")
        print(f"Function: {func_name}")
        print(f"Input: {args}")
        
        # Call function dynamically
        if func_name == "classify_text":
            result = classify_text(**args)
        elif func_name == "summarize_text":
            result = summarize_text(**args)
        else:
            print(f"❌ Function {func_name} not found!")
            continue
            
        print(f"Output: {result}")
        
        # Validate expected keys
        missing_keys = [key for key in expected_keys if key not in result]
        if missing_keys:
            print(f"⚠️  Missing keys: {missing_keys}")
        else:
            print("✅ All expected keys present")
```

### Step 3: Run Local Tests

```bash
cd "c:\Users\Betopia\Desktop\AI Structure"
python local_ai.py
```

### Step 4: Move to Production (`ai_logic.py`)

Once your functions work locally, move them to `ai_logic.py`:

```python
def classify_text(text, categories):
    """Production-ready text classification function"""
    try:
        # Your tested AI logic here
        result = {
            "category": "positive",
            "confidence": 0.95,
            "success": True,
            "message": "Classification completed successfully"
        }
        return result
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": "Classification failed"
        }
```

---

## 🔄 Development Workflow

### Daily Development Process

1. **Define Test Case** → Add to `data_format.py`
2. **Write Function** → Implement in `local_ai.py`
3. **Test Locally** → Run `python local_ai.py`
4. **Debug & Refine** → Iterate until tests pass
5. **Move to Production** → Copy working function to `ai_logic.py`
6. **Integrate** → Import in Django views/APIs

### Example Integration in Django

```python
# In your Django views.py
from .ai_logic import classify_text, summarize_text

def classify_view(request):
    if request.method == 'POST':
        text = request.data.get('text')
        categories = request.data.get('categories', [])
        
        # Call your tested AI function
        result = classify_text(text, categories)
        
        if result['success']:
            return Response(result, status=200)
        else:
            return Response(result, status=400)
```

---

## 🛠️ Advanced Usage

### Dynamic Function Calling

For multiple AI functions, use dynamic calling:

```python
# In local_ai.py
def run_tests():
    # Map function names to actual functions
    function_map = {
        'classify_text': classify_text,
        'summarize_text': summarize_text,
        'generate_keywords': generate_keywords,
    }
    
    for case in test_cases:
        func_name = case['function']
        args = case['input']
        
        if func_name in function_map:
            result = function_map[func_name](**args)
            print(f"✅ {case['name']}: {result}")
        else:
            print(f"❌ Function {func_name} not implemented")
```

### Error Handling Pattern

Consistent error handling across all AI functions:

```python
def ai_function_template(input_data):
    """Template for AI function with proper error handling"""
    try:
        # Validate input
        if not input_data:
            return {
                "success": False,
                "error": "Input data is required",
                "message": "Invalid input provided"
            }
        
        # Your AI logic here
        result = perform_ai_operation(input_data)
        
        return {
            "success": True,
            "data": result,
            "message": "Operation completed successfully"
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": "Operation failed"
        }
```

---

## 🧪 Testing Best Practices

### Comprehensive Test Cases

Create diverse test cases covering:

```python
test_cases = [
    # Happy path
    {
        "name": "normal_case",
        "function": "classify_text",
        "input": {"text": "Good product", "categories": ["pos", "neg"]},
        "expected_keys": ["category", "confidence", "success"]
    },
    # Edge cases
    {
        "name": "empty_text",
        "function": "classify_text",
        "input": {"text": "", "categories": ["pos", "neg"]},
        "expected_keys": ["success", "error"]
    },
    # Large input
    {
        "name": "long_text",
        "function": "classify_text",
        "input": {"text": "Very long text..." * 1000, "categories": ["pos", "neg"]},
        "expected_keys": ["category", "confidence", "success"]
    }
]
```

### Performance Testing

Add timing to your tests:

```python
import time

def run_performance_tests():
    for case in test_cases:
        start_time = time.time()
        
        # Run your function
        result = function_map[case['function']](**case['input'])
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        print(f"⏱️  {case['name']}: {execution_time:.2f}s")
```

---

## 📋 Best Practices

### Function Design
- **Pure Functions**: No side effects, same input = same output
- **Single Responsibility**: Each function does one thing well
- **Clear Input/Output**: Well-defined data structures
- **Error Handling**: Consistent error response format

### Testing Strategy
- **Test First**: Write test cases before implementing functions
- **Edge Cases**: Test empty inputs, large inputs, invalid data
- **Performance**: Monitor execution time for large datasets
- **Regression**: Re-run tests after any changes

### Code Organization
- **Modular**: Keep functions independent and reusable
- **Documentation**: Clear docstrings and comments
- **Version Control**: Track changes to test cases and functions
- **Dependencies**: Pin package versions in requirements.txt

---

## 🔧 Troubleshooting

### Common Issues

**Function not found error**:
```python
# Solution: Check function name spelling in data_format.py
# Ensure function exists in local_ai.py
```

**Missing expected keys**:
```python
# Solution: Check your function's return dictionary
# Ensure all expected_keys are included in the response
```

**Import errors**:
```python
# Solution: Check file paths and Python path
# Ensure data_format.py is in the same directory
```

### Debug Mode

Add debug output to your functions:

```python
def classify_text(text, categories, debug=False):
    if debug:
        print(f"Input text length: {len(text)}")
        print(f"Categories: {categories}")
    
    # Your AI logic here
    result = {"category": "positive", "confidence": 0.95}
    
    if debug:
        print(f"Result: {result}")
    
    return result
```

---

## 🚀 Ready to Deploy?

Once your AI functions are working perfectly in `local_ai.py`:

1. **Copy to `ai_logic.py`** - Move your tested functions
2. **Add Error Handling** - Wrap in try-catch blocks
3. **Import in Views** - Use in Django/Flask/FastAPI
4. **Monitor Performance** - Track execution time in production
5. **Update Tests** - Keep test cases current with new features

This structure ensures your AI logic is **reliable, testable, and production-ready**! 🎉