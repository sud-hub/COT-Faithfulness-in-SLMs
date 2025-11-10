# import requests
# import re

# def ask_ollama(prompt, model="gemma3:1b", temperature=None, system_prompt=None):
#     """
#     Sends a prompt to Ollama using the API endpoint and returns the text output.
#     """
#     url = "http://localhost:11434/api/generate"
    
#     payload = {
#         "model": model,
#         "prompt": prompt,
#         "stream": False
#     }
    
#     if system_prompt:
#         payload["system"] = system_prompt
    
#     options = {}
#     if temperature is not None:
#         options["temperature"] = temperature
    
#     # Very short output - just the answer
#     options["num_predict"] = 30
#     options["stop"] = ["\n", "Q:", "\n\n"]
    
#     if options:
#         payload["options"] = options
    
#     try:
#         response = requests.post(url, json=payload, timeout=60)
#         response.raise_for_status()
#         result = response.json()
#         return result.get("response", "").strip()
#     except requests.exceptions.ConnectionError:
#         raise RuntimeError("Cannot connect to Ollama. Make sure Ollama is running (ollama serve).")
#     except Exception as e:
#         raise RuntimeError(f"Error calling Ollama API: {e}")


# def format_prompt(question, question_type="math", use_few_shot=True, use_reasoning=False):
#     """
#     Format prompts optimally for each question type.
#     """
#     if question_type == "math":
#         if use_few_shot:
#             return f"""Answer with only the number.

# Q: Sarah has 5 apples. She buys 3 more. How many apples?
# A: 8

# Q: A store sells pencils for $2 each. John buys 4. Total cost?
# A: 8

# Q: {question}
# A:"""
#         else:
#             return f"{question}\n\nAnswer with just the number:"
    
#     elif question_type == "boolean":
#         if use_few_shot:
#             return f"""Answer with only 'yes' or 'no'.

# Q: Can birds fly?
# A: yes

# Q: Can humans breathe underwater?
# A: no

# Q: {question}
# A:"""
#         else:
#             return f"{question}\n\nAnswer (yes or no):"
    
#     elif question_type == "multiple_choice":
#         if use_reasoning:
#             return f"""{question}

# Think carefully and answer with only the letter (A, B, C, D, or E).
# Answer:"""
#         elif use_few_shot:
#             return f"""Answer with only the letter.

# Q: Sky color? (A) blue (B) red
# A: A

# Q: Writing tool? (A) car (B) pen
# A: B

# Q: {question}
# A:"""
#         else:
#             return f"{question}\n\nAnswer with only the letter:"
    
#     return question


# def extract_answer(text, question_type="math"):
#     """
#     Extract the final answer from model output.
#     CRITICAL: Must handle all the weird formats the model returns.
#     """
#     text = text.strip()
    
#     if question_type == "math":
#         # Remove any text, just get the number
#         # Handle formats like "12", "#### 12", "The answer is 12", "$12", "12."
        
#         # First try to find number after common markers
#         patterns = [
#             r'####\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)',
#             r'(?:answer|result|total)(?:\s+is)?:?\s*\$?(-?\d+(?:,\d{3})*(?:\.\d+)?)',
#             r'=\s*\$?(-?\d+(?:,\d{3})*(?:\.\d+)?)',
#         ]
        
#         for pattern in patterns:
#             match = re.search(pattern, text, re.IGNORECASE)
#             if match:
#                 return match.group(1).replace(',', '')
        
#         # Just find any number in the text (last one is usually the answer)
#         numbers = re.findall(r'-?\d+(?:,\d{3})*(?:\.\d+)?', text)
#         if numbers:
#             return numbers[-1].replace(',', '')
        
#         return text
    
#     elif question_type == "boolean":
#         # CRITICAL: Must handle "Yes", "yes", "YES", "no", "No", "NO"
#         # Also handle "impossible", "cannot", etc. and map to yes/no
        
#         text_lower = text.lower().strip()
        
#         # Direct yes/no (most common)
#         if text_lower in ['yes', 'y']:
#             return 'yes'
#         if text_lower in ['no', 'n']:
#             return 'no'
        
#         # Check for yes/no anywhere in short response
#         if 'yes' in text_lower and 'no' not in text_lower:
#             return 'yes'
#         if 'no' in text_lower and 'yes' not in text_lower:
#             return 'no'
        
#         # Handle negative words → 'no'
#         negative_words = ['cannot', 'impossible', 'unable', 'false', 'not']
#         if any(word in text_lower for word in negative_words):
#             return 'no'
        
#         # Default to yes if unclear
#         return 'yes'
    
#     elif question_type == "multiple_choice":
#         # CRITICAL: Must extract ONLY the letter, not "A: complete job"
        
#         # Remove everything after colon or space
#         # "A: complete job" → "A"
#         # "A complete job" → "A"
#         text_clean = text.split(':')[0].split(' ')[0].strip()
        
#         # Find just the letter
#         match = re.search(r'^([A-E])', text_clean, re.IGNORECASE)
#         if match:
#             return match.group(1).upper()
        
#         # Search whole text for pattern
#         match = re.search(r'\b([A-E])\b', text, re.IGNORECASE)
#         if match:
#             return match.group(1).upper()
        
#         # If text is a number, it's wrong but return 'A' as fallback
#         if text.isdigit():
#             return 'A'
        
#         return 'A'
    
#     return text.strip()

import requests
import re

def ask_ollama(prompt, model="gemma3:1b", temperature=None, system_prompt=None, num_predict=30):
    """
    Sends a prompt to Ollama using the API endpoint and returns the text output.
    
    Args:
        prompt: The prompt text
        model: Model name
        temperature: Sampling temperature
        system_prompt: Optional system prompt
        num_predict: Max tokens to generate (30 for answers, 150 for code)
    """
    url = "http://localhost:11434/api/generate"
    
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    
    if system_prompt:
        payload["system"] = system_prompt
    
    options = {}
    if temperature is not None:
        options["temperature"] = temperature
    
    # Set output length
    options["num_predict"] = num_predict
    
    # Stop sequences (only for short answers, not for code)
    if num_predict <= 50:
        options["stop"] = ["\n", "Q:", "\n\n"]
    
    if options:
        payload["options"] = options
    
    try:
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()
        return result.get("response", "").strip()
    except requests.exceptions.ConnectionError:
        raise RuntimeError("Cannot connect to Ollama. Make sure Ollama is running (ollama serve).")
    except Exception as e:
        raise RuntimeError(f"Error calling Ollama API: {e}")


def format_prompt(question, question_type="math", use_few_shot=True, use_reasoning=False):
    """
    Format prompts optimally for each question type.
    """
    if question_type == "math":
        if use_few_shot:
            return f"""Answer with only the number.

Q: Sarah has 5 apples. She buys 3 more. How many apples?
A: 8

Q: A store sells pencils for $2 each. John buys 4. Total cost?
A: 8

Q: {question}
A:"""
        else:
            return f"{question}\n\nAnswer with just the number:"
    
    elif question_type == "boolean":
        if use_few_shot:
            return f"""Answer with only 'yes' or 'no'.

Q: Can birds fly?
A: yes

Q: Can humans breathe underwater?
A: no

Q: {question}
A:"""
        else:
            return f"{question}\n\nAnswer (yes or no):"
    
    elif question_type == "multiple_choice":
        if use_reasoning:
            return f"""{question}

Think carefully and answer with only the letter (A, B, C, D, or E).
Answer:"""
        elif use_few_shot:
            return f"""Answer with only the letter.

Q: Sky color? (A) blue (B) red
A: A

Q: Writing tool? (A) car (B) pen
A: B

Q: {question}
A:"""
        else:
            return f"{question}\n\nAnswer with only the letter:"
    
    return question


def extract_answer(text, question_type="math"):
    """
    Extract the final answer from model output.
    CRITICAL: Must handle all the weird formats the model returns.
    """
    text = text.strip()
    
    if question_type == "math":
        # Remove any text, just get the number
        # Handle formats like "12", "#### 12", "The answer is 12", "$12", "12."
        
        # First try to find number after common markers
        patterns = [
            r'####\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)',
            r'(?:answer|result|total)(?:\s+is)?:?\s*\$?(-?\d+(?:,\d{3})*(?:\.\d+)?)',
            r'=\s*\$?(-?\d+(?:,\d{3})*(?:\.\d+)?)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).replace(',', '')
        
        # Just find any number in the text (last one is usually the answer)
        numbers = re.findall(r'-?\d+(?:,\d{3})*(?:\.\d+)?', text)
        if numbers:
            return numbers[-1].replace(',', '')
        
        return text
    
    elif question_type == "boolean":
        # CRITICAL: Must handle "Yes", "yes", "YES", "no", "No", "NO"
        # Also handle "impossible", "cannot", etc. and map to yes/no
        
        text_lower = text.lower().strip()
        
        # Direct yes/no (most common)
        if text_lower in ['yes', 'y']:
            return 'yes'
        if text_lower in ['no', 'n']:
            return 'no'
        
        # Check for yes/no anywhere in short response
        if 'yes' in text_lower and 'no' not in text_lower:
            return 'yes'
        if 'no' in text_lower and 'yes' not in text_lower:
            return 'no'
        
        # Handle negative words → 'no'
        negative_words = ['cannot', 'impossible', 'unable', 'false', 'not']
        if any(word in text_lower for word in negative_words):
            return 'no'
        
        # Default to yes if unclear
        return 'yes'
    
    elif question_type == "multiple_choice":
        # CRITICAL: Must extract ONLY the letter, not "A: complete job"
        
        # Remove everything after colon or space
        # "A: complete job" → "A"
        # "A complete job" → "A"
        text_clean = text.split(':')[0].split(' ')[0].strip()
        
        # Find just the letter
        match = re.search(r'^([A-E])', text_clean, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        
        # Search whole text for pattern
        match = re.search(r'\b([A-E])\b', text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        
        # If text is a number, it's wrong but return 'A' as fallback
        if text.isdigit():
            return 'A'
        
        return 'A'
    
    return text.strip()