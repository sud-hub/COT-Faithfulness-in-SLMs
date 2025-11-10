"""
Program of Thoughts (PoT) - Generate and execute Python code to solve math problems
This is much more effective than pure text reasoning for GSM8K
"""

from ollama_utils import ask_ollama
import re
import traceback

def generate_python_code(question, model="gemma3:1b"):
    """
    Ask the model to generate Python code to solve the math problem
    """
    prompt = f"""Write Python code to solve this math problem. Use simple arithmetic operations and print only the final numerical answer.

Example 1:
Problem: Sarah has 5 apples. She buys 3 more. How many apples does she have?
Code:
initial = 5
bought = 3
total = initial + bought
print(total)

Example 2:
Problem: A store sells pencils for $2 each. John buys 4 pencils. How much does he spend?
Code:
price_per_pencil = 2
quantity = 4
total_cost = price_per_pencil * quantity
print(total_cost)

Now solve this:
Problem: {question}
Code:"""
    
    try:
        # Get code from model with higher token limit for code generation
        response = ask_ollama(prompt, model=model, temperature=0, num_predict=150)
        return response
    except Exception as e:
        print(f"Error generating code: {e}")
        return None


def extract_python_code(text):
    """
    Extract Python code from model response
    Handles various formats: code blocks, plain code, etc.
    """
    # Remove markdown code blocks if present
    text = re.sub(r'```python\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    
    # Remove any explanatory text before/after code
    lines = text.split('\n')
    code_lines = []
    in_code = False
    
    for line in lines:
        stripped = line.strip()
        
        # Skip empty lines at start
        if not stripped and not in_code:
            continue
        
        # Detect code (lines with =, operators, print, etc.)
        if any(keyword in stripped for keyword in ['=', 'print(', '+', '-', '*', '/', 'def ', 'for ', 'if ']):
            in_code = True
            code_lines.append(line)
        elif in_code and stripped:
            # If we're in code and see explanatory text, stop
            if stripped.startswith('#'):
                code_lines.append(line)
            elif any(char.isalpha() for char in stripped) and '=' not in stripped and 'print' not in stripped:
                break
            else:
                code_lines.append(line)
        elif in_code and not stripped:
            # Empty line in code
            code_lines.append(line)
    
    return '\n'.join(code_lines).strip()


def execute_python_code(code, timeout=5):
    """
    Safely execute Python code and capture output
    
    Args:
        code: Python code string
        timeout: Execution timeout in seconds
    
    Returns:
        (success, result) tuple
    """
    import io
    import sys
    from contextlib import redirect_stdout
    
    # Create a restricted namespace for execution
    namespace = {
        '__builtins__': {
            'print': print,
            'len': len,
            'range': range,
            'sum': sum,
            'max': max,
            'min': min,
            'abs': abs,
            'round': round,
            'int': int,
            'float': float,
            'str': str,
        }
    }
    
    # Capture output
    output = io.StringIO()
    
    try:
        with redirect_stdout(output):
            # Execute with timeout (simple version, no actual timeout enforcement)
            exec(code, namespace)
        
        result = output.getvalue().strip()
        
        # Extract number from output
        numbers = re.findall(r'-?\d+\.?\d*', result)
        if numbers:
            return True, numbers[-1]  # Return last number printed
        else:
            return True, result
    
    except Exception as e:
        return False, str(e)


def program_of_thought(question, model="gemma3:1b", max_retries=2):
    """
    Program of Thought: Generate Python code and execute it to get answer
    
    Args:
        question: Math word problem
        model: LLM model to use
        max_retries: Number of times to retry if code fails
    
    Returns:
        Final numerical answer as string
    """
    for attempt in range(max_retries):
        # Generate code
        code_response = generate_python_code(question, model=model)
        
        if not code_response:
            continue
        
        # Extract clean code
        code = extract_python_code(code_response)
        
        if not code:
            continue
        
        # Execute code
        success, result = execute_python_code(code)
        
        if success and result:
            # Clean result
            result = result.replace(',', '').strip()
            return result
        
        # If failed and we have retries left, try again
        if attempt < max_retries - 1:
            continue
    
    # If all attempts failed, return empty
    return ""


def pot_self_consistency(question, model="gemma3:1b", samples=3):
    """
    Program of Thought with Self-Consistency
    Generate multiple programs and take majority vote
    
    Args:
        question: Math word problem
        model: LLM model
        samples: Number of programs to generate
    
    Returns:
        Most common answer
    """
    from collections import Counter
    
    answers = []
    
    for i in range(samples):
        answer = program_of_thought(question, model=model, max_retries=1)
        if answer:
            answers.append(answer)
    
    if not answers:
        return ""
    
    # Majority vote
    answer_counts = Counter(answers)
    return answer_counts.most_common(1)[0][0]


# Fallback: Generate code for simple arithmetic directly without LLM
def simple_arithmetic_solver(question):
    """
    Fallback solver for very simple arithmetic that can be extracted directly
    Uses pattern matching to extract numbers and operations
    """
    # Extract all numbers from question
    numbers = re.findall(r'\d+\.?\d*', question)
    
    if len(numbers) < 2:
        return None
    
    # Try to detect operation from keywords
    question_lower = question.lower()
    
    try:
        nums = [float(n) for n in numbers[:2]]
        
        if any(word in question_lower for word in ['total', 'sum', 'add', 'plus', 'more', 'altogether']):
            return str(int(nums[0] + nums[1]))
        elif any(word in question_lower for word in ['difference', 'subtract', 'minus', 'less', 'fewer', 'left']):
            return str(int(nums[0] - nums[1]))
        elif any(word in question_lower for word in ['product', 'multiply', 'times', 'each']):
            return str(int(nums[0] * nums[1]))
        elif any(word in question_lower for word in ['divide', 'per', 'split', 'share']):
            if nums[1] != 0:
                return str(int(nums[0] / nums[1]))
    except:
        pass
    
    return None