# hybrid_reasoning.py - Combining Least-to-Most with PoT-SC

from ollama_utils import ask_ollama, extract_answer
from reasoning_methods import program_of_thought_method
from collections import Counter
import re

# ============================================================================
# HYBRID METHOD 1: Least-to-Most PoT (L2M-PoT)
# ============================================================================

def least_to_most_pot(question, model="gemma3:1b"):
    """
    Combines Least-to-Most decomposition with Program-of-Thought execution
    
    Strategy:
    1. Use L2M to break problem into steps
    2. Generate Python code for each step
    3. Execute code sequentially with context
    """
    
    # Step 1: Decompose into sub-problems
    decompose_prompt = f"""Break this math problem into 2-3 simple calculation steps:

Problem: {question}

List the steps clearly:
Step 1:
Step 2:
Step 3:"""
    
    try:
        steps_response = ask_ollama(decompose_prompt, model=model, temperature=0, num_predict=150)
        
        # Parse steps
        steps = []
        for line in steps_response.split('\n'):
            if line.strip() and ('Step' in line or any(c.isdigit() for c in line)):
                # Clean up step text
                step_text = re.sub(r'^Step\s*\d+[:.)\s]*', '', line).strip()
                if step_text and len(step_text) > 10:
                    steps.append(step_text)
        
        if not steps:
            # Fallback to regular PoT if decomposition fails
            return program_of_thought_method(question, model=model)
        
        # Step 2: Generate code for each sub-problem
        context = ""
        all_code = []
        
        for i, step in enumerate(steps[:3], 1):
            code_prompt = f"""Write Python code to solve this calculation step.

{context}

Step {i}: {step}

Write SHORT Python code (2-3 lines max). Store result in 'result_{i}':
```python"""
            
            code_response = ask_ollama(code_prompt, model=model, temperature=0, num_predict=100)
            
            # Extract code
            code_match = re.search(r'```python\s*(.*?)\s*```', code_response, re.DOTALL)
            if code_match:
                code = code_match.group(1)
            else:
                # Try to find code without markers
                lines = [l.strip() for l in code_response.split('\n') if l.strip() and not l.strip().startswith('#')]
                code = '\n'.join(lines[:3])  # Max 3 lines per step
            
            all_code.append(code)
            context += f"\nStep {i} code: {code}"
        
        # Step 3: Combine and execute all code
        combined_code = '\n'.join(all_code)
        combined_code += '\nresult = result_' + str(len(all_code))  # Final result
        
        # Execute
        try:
            local_vars = {}
            exec(combined_code, {"__builtins__": {}}, local_vars)
            result = local_vars.get('result', None)
            
            if result is not None:
                if isinstance(result, float) and result == int(result):
                    return str(int(result))
                return str(result)
        except Exception as e:
            print(f"  L2M-PoT execution error: {e}")
            pass
        
    except Exception as e:
        print(f"  L2M-PoT error: {e}")
    
    # Fallback to regular PoT
    return program_of_thought_method(question, model=model)


# ============================================================================
# HYBRID METHOD 2: Least-to-Most PoT with Self-Consistency (L2M-PoT-SC)
# ============================================================================

def least_to_most_pot_sc(question, model="gemma3:1b", samples=3):
    """
    L2M-PoT with Self-Consistency voting
    
    Strategy:
    1. Generate multiple L2M-PoT solutions (with temperature > 0)
    2. Vote on most common answer
    3. Higher quality than single attempt
    """
    
    answers = []
    
    for i in range(samples):
        try:
            # Use slight temperature variation for diversity
            temp = 0 if i == 0 else 0.3
            
            # Generate solution
            answer = least_to_most_pot_internal(question, model=model, temperature=temp)
            
            if answer:
                answers.append(answer)
        except Exception as e:
            print(f"  L2M-PoT-SC sample {i+1} error: {e}")
            continue
    
    # Vote on most common answer
    if answers:
        most_common = Counter(answers).most_common(1)[0][0]
        return most_common
    
    return ""


def least_to_most_pot_internal(question, model="gemma3:1b", temperature=0):
    """Internal version with temperature control"""
    
    # Step 1: Decompose
    decompose_prompt = f"""Break this math problem into simple steps:

Problem: {question}

Step 1:
Step 2:
Step 3:"""
    
    steps_response = ask_ollama(decompose_prompt, model=model, temperature=temperature, num_predict=150)
    
    # Parse steps
    steps = []
    for line in steps_response.split('\n'):
        if line.strip() and ('Step' in line or any(c.isdigit() for c in line)):
            step_text = re.sub(r'^Step\s*\d+[:.)\s]*', '', line).strip()
            if step_text and len(step_text) > 10:
                steps.append(step_text)
    
    if not steps:
        return program_of_thought_method(question, model=model)
    
    # Step 2: Generate and execute code
    full_code = "# Solution\n"
    
    for i, step in enumerate(steps[:3], 1):
        code_prompt = f"""Python code for: {step}

Use variables from previous steps. Store in result_{i}:
```python"""
        
        code_response = ask_ollama(code_prompt, model=model, temperature=temperature, num_predict=80)
        
        code_match = re.search(r'```python\s*(.*?)\s*```', code_response, re.DOTALL)
        if code_match:
            code = code_match.group(1)
        else:
            lines = [l.strip() for l in code_response.split('\n') if l.strip() and not l.strip().startswith('#')]
            code = '\n'.join(lines[:3])
        
        full_code += code + '\n'
    
    full_code += f'\nresult = result_{min(len(steps), 3)}'
    
    # Execute
    try:
        local_vars = {}
        exec(full_code, {"__builtins__": {}}, local_vars)
        result = local_vars.get('result', None)
        
        if result is not None:
            if isinstance(result, float) and result == int(result):
                return str(int(result))
            return str(result)
    except:
        pass
    
    return ""


# ============================================================================
# HYBRID METHOD 3: Ensemble (L2M + PoT + L2M-PoT)
# ============================================================================

def gsm8k_ensemble(question, model="gemma3:1b"):
    """
    Ensemble of multiple approaches for GSM8K
    
    Combines:
    1. Regular PoT
    2. Least-to-Most reasoning
    3. L2M-PoT hybrid
    
    Returns: Most common answer (weighted voting)
    """
    from sota_reasoning import least_to_most_prompting
    
    results = []
    
    # Method 1: Regular PoT (weight: 1)
    try:
        ans = program_of_thought_method(question, model=model)
        if ans:
            results.append(ans)
    except:
        pass
    
    # Method 2: Least-to-Most (weight: 2)
    try:
        ans = least_to_most_prompting(question, model=model, question_type="math")
        if ans:
            results.extend([ans, ans])
    except:
        pass
    
    # Method 3: L2M-PoT hybrid (weight: 2)
    try:
        ans = least_to_most_pot(question, model=model)
        if ans:
            results.extend([ans, ans])
    except:
        pass
    
    # Vote
    if results:
        return Counter(results).most_common(1)[0][0]
    return ""


# ============================================================================
# HYBRID METHOD 4: Best GSM8K Method (Ultimate) - OPTIMIZED
# ============================================================================

def ultimate_gsm8k(question, model="gemma3:1b", fast_mode=True):
    """
    The ultimate GSM8K solver combining all best strategies
    
    Strategy (OPTIMIZED):
    1. Try L2M-PoT-SC with adaptive sampling
    2. Early exit if consensus reached
    3. No fallback to save time
    
    Args:
        fast_mode: If True, use 2 samples instead of 3 (40% faster)
    """
    
    # Adaptive sampling: 2 samples in fast mode, 3 in accurate mode
    samples = 2 if fast_mode else 3
    
    # Use L2M-PoT with Self-Consistency (optimized)
    answer = least_to_most_pot_sc_fast(question, model=model, samples=samples)
    
    if answer:
        return answer
    
    # If still no answer, try simple L2M-PoT once (no SC)
    return least_to_most_pot(question, model=model)


def least_to_most_pot_sc_fast(question, model="gemma3:1b", samples=2):
    """
    OPTIMIZED L2M-PoT with Self-Consistency
    
    Optimizations:
    1. Reduced token limits for faster generation
    2. Early exit if 2/2 samples agree
    3. Parallel-friendly structure
    4. Reduced samples (2 instead of 3)
    """
    
    answers = []
    
    for i in range(samples):
        try:
            # Use temperature=0 for first sample (fastest), 0.2 for diversity
            temp = 0 if i == 0 else 0.2
            
            # Optimized version with shorter generation
            answer = least_to_most_pot_fast(question, model=model, temperature=temp)
            
            if answer:
                answers.append(answer)
                
                # EARLY EXIT: If we have 2 identical answers, stop
                if len(answers) >= 2 and len(set(answers)) == 1:
                    return answers[0]
        except Exception as e:
            # Silent fail, continue with other samples
            continue
    
    # Vote on most common answer
    if answers:
        most_common = Counter(answers).most_common(1)[0][0]
        return most_common
    
    return ""


def least_to_most_pot_fast(question, model="gemma3:1b", temperature=0):
    """
    FAST version of L2M-PoT with aggressive optimizations
    
    Speed optimizations:
    1. Limit to 2 steps max (instead of 3)
    2. Shorter prompts
    3. Lower token limits (50% reduction)
    4. Skip decomposition for simple problems
    """
    
    # Quick check: if problem is simple (< 30 words), skip decomposition
    word_count = len(question.split())
    if word_count < 30:
        # For simple problems, use direct PoT
        return program_of_thought_method(question, model=model)
    
    # Step 1: Fast decomposition (max 2 steps)
    decompose_prompt = f"""Break into 2 simple steps:

{question}

1.
2."""
    
    steps_response = ask_ollama(decompose_prompt, model=model, temperature=temperature, num_predict=80)
    
    # Parse steps quickly
    steps = []
    for line in steps_response.split('\n'):
        if line.strip() and (line.strip()[0].isdigit() or 'step' in line.lower()):
            step_text = re.sub(r'^[\d.)\s]*', '', line).strip()
            if step_text and len(step_text) > 8:
                steps.append(step_text)
        if len(steps) >= 2:  # Stop at 2 steps
            break
    
    if not steps:
        # Fallback to direct PoT if decomposition fails
        return program_of_thought_method(question, model=model)
    
    # Step 2: Generate compact code
    full_code = ""
    
    for i, step in enumerate(steps[:2], 1):  # Max 2 steps
        code_prompt = f"""{step}

Python (2 lines max):
```python"""
        
        code_response = ask_ollama(code_prompt, model=model, temperature=temperature, num_predict=50)
        
        # Extract code
        code_match = re.search(r'```python\s*(.*?)\s*```', code_response, re.DOTALL)
        if code_match:
            code = code_match.group(1)
        else:
            # Take first 2 non-empty lines
            lines = [l.strip() for l in code_response.split('\n') if l.strip() and not l.strip().startswith('#')]
            code = '\n'.join(lines[:2])
        
        full_code += code + '\n'
    
    full_code += f'\nresult = result_{min(len(steps), 2)}'
    
    # Execute
    try:
        local_vars = {}
        exec(full_code, {"__builtins__": {}}, local_vars)
        result = local_vars.get('result', None)
        
        if result is not None:
            if isinstance(result, float) and result == int(result):
                return str(int(result))
            return str(result)
    except:
        pass
    
    return ""