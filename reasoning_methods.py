# from ollama_utils import ask_ollama, extract_answer, format_prompt
# from collections import Counter

# def self_consistency(prompt, model="gemma3:1b", temperature=0.7, samples=5, question_type="math"):
#     """
#     Run self-consistency: multiple stochastic samples and majority vote
    
#     Args:
#         prompt: The question prompt (already formatted)
#         model: Model name
#         temperature: Sampling temperature (higher = more diverse)
#         samples: Number of samples to generate
#         question_type: Type of question for answer extraction
    
#     Returns:
#         Most common answer after extraction
#     """
#     raw_outputs = []
#     extracted_answers = []
    
#     for i in range(samples):
#         try:
#             out = ask_ollama(prompt, model=model, temperature=temperature)
#             raw_outputs.append(out)
            
#             # Extract the actual answer
#             answer = extract_answer(out, question_type=question_type)
#             extracted_answers.append(answer)
#         except Exception as e:
#             print(f"Error on sample {i+1}/{samples}: {e}")
#             continue
    
#     if not extracted_answers:
#         return ""
    
#     # Majority vote
#     answer_counts = Counter(extracted_answers)
#     most_common_answer = answer_counts.most_common(1)[0][0]
    
#     # Optional: print distribution for debugging
#     # print(f"Answer distribution: {dict(answer_counts)}")
    
#     return most_common_answer


# def chain_of_thought(prompt, model="gemma3:1b", question_type="math"):
#     """
#     Chain of thought prompting - explicitly ask model to think step by step
#     Note: For multiple choice, we already use reasoning in the prompt, so this
#     might not add much value unless we want even more explicit step-by-step.
#     """
#     # If prompt already has reasoning instruction, use as-is
#     if "analyze" in prompt.lower() or "step by step" in prompt.lower():
#         cot_prompt = prompt
#     else:
#         # Add CoT instruction
#         cot_prompt = f"{prompt}\n\nLet's think step by step and then provide the final answer."
    
#     output = ask_ollama(cot_prompt, model=model, temperature=0)
#     return extract_answer(output, question_type=question_type)

from ollama_utils import ask_ollama, extract_answer
from collections import Counter

def self_consistency(prompt, model="gemma3:1b", temperature=0.7, samples=5, question_type="math"):
    """
    Run self-consistency: multiple stochastic samples and majority vote
    
    Args:
        prompt: The question prompt (already formatted with format_prompt)
        model: Model name
        temperature: Sampling temperature (higher = more diverse)
        samples: Number of samples to generate
        question_type: Type of question for answer extraction
    
    Returns:
        Most common answer after extraction
    """
    extracted_answers = []
    
    for i in range(samples):
        try:
            out = ask_ollama(prompt, model=model, temperature=temperature)
            
            # Extract the actual answer
            answer = extract_answer(out, question_type=question_type)
            extracted_answers.append(answer)
        except Exception as e:
            print(f"Error on sample {i+1}/{samples}: {e}")
            continue
    
    if not extracted_answers:
        return ""
    
    # Majority vote
    answer_counts = Counter(extracted_answers)
    most_common_answer = answer_counts.most_common(1)[0][0]
    
    return most_common_answer


def chain_of_thought(prompt, model="gemma3:1b", question_type="math"):
    """
    Chain of thought prompting - explicitly ask model to think step by step
    Note: For multiple choice, we already use reasoning in the prompt via
    format_prompt(use_reasoning=True), so we don't duplicate instructions.
    """
    # If prompt already has reasoning instruction, use as-is
    if "analyze" in prompt.lower() or "step by step" in prompt.lower():
        cot_prompt = prompt
    else:
        # Add CoT instruction
        cot_prompt = f"{prompt}\n\nLet's think step by step and then provide the final answer."
    
    output = ask_ollama(cot_prompt, model=model, temperature=0)
    return extract_answer(output, question_type=question_type)


def program_of_thought_method(question, model="gemma3:1b"):
    """
    Program of Thought (PoT) - Generate and execute Python code to solve math
    This is MUCH better for GSM8K than pure text reasoning!
    
    Args:
        question: Raw question text (not pre-formatted)
        model: Model to use for code generation
    
    Returns:
        Numerical answer as string
    """
    # Import here to avoid circular dependency
    from pot_reasoning import program_of_thought
    return program_of_thought(question, model=model, max_retries=2)


def pot_with_self_consistency(question, model="gemma3:1b", samples=3):
    """
    PoT + Self-Consistency: Generate multiple programs and vote
    Best method for GSM8K!
    
    Args:
        question: Raw question text
        model: Model to use
        samples: Number of programs to generate
    
    Returns:
        Most common answer
    """
    from pot_reasoning import pot_self_consistency
    return pot_self_consistency(question, model=model, samples=samples)