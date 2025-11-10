#sota_reasoning.py
"""
State-of-the-art reasoning methods from recent research (2023-2024)
"""

from ollama_utils import ask_ollama, extract_answer
from collections import Counter
import re

# ============================================================================
# 1. LEAST-TO-MOST PROMPTING (Zhou et al., 2023)
# ============================================================================

def least_to_most_prompting(question, model="gemma3:1b", question_type="math"):
    """
    Break complex problems into simpler sub-problems
    Paper: "Least-to-Most Prompting Enables Complex Reasoning"
    """
    
    # Step 1: Decompose into sub-questions
    decompose_prompt = f"""Break this problem into 2-3 simpler steps:

Problem: {question}

Step 1:
Step 2:
Step 3:"""
    
    try:
        steps_response = ask_ollama(decompose_prompt, model=model, temperature=0, num_predict=100)
        steps = [s.strip() for s in steps_response.split('\n') if s.strip()]
        
        # Step 2: Solve each sub-problem
        solutions = []
        context = ""
        
        for step in steps[:3]:  # Max 3 steps
            if not step:
                continue
                
            solve_prompt = f"""Given: {context}

Question: {step}

Answer briefly:"""
            
            sub_answer = ask_ollama(solve_prompt, model=model, temperature=0, num_predict=50)
            solutions.append(sub_answer)
            context += f"\n{step} → {sub_answer}"
        
        # Step 3: Combine for final answer
        final_prompt = f"""Problem: {question}

Steps solved:
{context}

Final answer (just the number/letter):"""
        
        final_response = ask_ollama(final_prompt, model=model, temperature=0, num_predict=30)
        return extract_answer(final_response, question_type=question_type)
        
    except Exception as e:
        print(f"Least-to-most error: {e}")
        return ""


# ============================================================================
# 2. SELF-REFINE (Madaan et al., 2023)
# ============================================================================

def self_refine(question, model="gemma3:1b", question_type="math", iterations=2):
    """
    Generate answer, critique it, refine it
    Paper: "Self-Refine: Iterative Refinement with Self-Feedback"
    """
    
    # Initial answer
    if question_type == "math":
        prompt = f"{question}\n\nSolve step by step. Final answer:"
    else:
        prompt = f"{question}\n\nAnswer:"
    
    answer = ask_ollama(prompt, model=model, temperature=0, num_predict=100)
    
    # Refine iterations
    for i in range(iterations):
        # Critique
        critique_prompt = f"""Question: {question}
Current answer: {answer}

What could be wrong with this answer? Be specific:"""
        
        critique = ask_ollama(critique_prompt, model=model, temperature=0.3, num_predict=80)
        
        # Refine
        refine_prompt = f"""Question: {question}
Previous answer: {answer}
Issue found: {critique}

Provide a better answer:"""
        
        answer = ask_ollama(refine_prompt, model=model, temperature=0, num_predict=100)
    
    return extract_answer(answer, question_type=question_type)


# ============================================================================
# 3. PLAN-AND-SOLVE (Wang et al., 2023)
# ============================================================================

def plan_and_solve(question, model="gemma3:1b", question_type="math"):
    """
    First make a plan, then execute it
    Paper: "Plan-and-Solve Prompting"
    """
    
    # Step 1: Make a plan
    plan_prompt = f"""Problem: {question}

Make a step-by-step plan to solve this (3-4 steps):
1."""
    
    plan = ask_ollama(plan_prompt, model=model, temperature=0, num_predict=100)
    
    # Step 2: Execute the plan
    solve_prompt = f"""Problem: {question}

Plan:
{plan}

Now execute this plan and give the final answer:"""
    
    solution = ask_ollama(solve_prompt, model=model, temperature=0, num_predict=100)
    
    return extract_answer(solution, question_type=question_type)


# ============================================================================
# 4. ANALOGICAL PROMPTING (Yasunaga et al., 2024)
# ============================================================================

def analogical_prompting(question, model="gemma3:1b", question_type="multiple_choice"):
    """
    Generate similar examples, learn from them
    Paper: "Large Language Models as Analogical Reasoners"
    """
    
    # Generate analogous examples
    analogy_prompt = f"""Think of a simpler, similar question to: {question}

Simpler question:"""
    
    similar_q = ask_ollama(analogy_prompt, model=model, temperature=0.5, num_predict=80)
    
    # Solve the simpler question
    solve_similar = f"""{similar_q}

Answer:"""
    
    similar_ans = ask_ollama(solve_similar, model=model, temperature=0, num_predict=30)
    
    # Apply learning to original
    apply_prompt = f"""I solved a similar problem:
Q: {similar_q}
A: {similar_ans}

Now solve the original:
Q: {question}

Answer with just the letter:"""
    
    final_ans = ask_ollama(apply_prompt, model=model, temperature=0, num_predict=30)
    
    return extract_answer(final_ans, question_type=question_type)


# ============================================================================
# 5. METACOGNITIVE PROMPTING (Chen et al., 2024)
# ============================================================================

def metacognitive_prompting(question, model="gemma3:1b", question_type="multiple_choice"):
    """
    Make model think about its own reasoning process
    """
    
    prompt = f"""{question}

Before answering, think about:
1. What type of knowledge is needed?
2. What's the most reliable way to approach this?
3. What might cause me to make a mistake?

Now answer with the letter:"""
    
    response = ask_ollama(prompt, model=model, temperature=0, num_predict=150)
    return extract_answer(response, question_type=question_type)


# ============================================================================
# 6. TREE-OF-THOUGHTS LIGHT (Yao et al., 2023 - simplified)
# ============================================================================

def tree_of_thoughts_light(question, model="gemma3:1b", question_type="multiple_choice", branches=3):
    """
    Explore multiple reasoning paths, select best
    Simplified version of Tree-of-Thoughts
    """
    
    # Generate multiple reasoning paths
    paths = []
    
    reasoning_prompts = [
        f"{question}\n\nApproach 1 - Eliminate wrong answers first:\n",
        f"{question}\n\nApproach 2 - Think of real-world examples:\n",
        f"{question}\n\nApproach 3 - Consider what makes most sense:\n"
    ]
    
    for prompt in reasoning_prompts[:branches]:
        try:
            response = ask_ollama(prompt, model=model, temperature=0.3, num_predict=100)
            answer = extract_answer(response, question_type=question_type)
            if answer:
                paths.append(answer)
        except:
            continue
    
    # Vote on best path
    if paths:
        return Counter(paths).most_common(1)[0][0]
    return ""


# ============================================================================
# 7. PROGRESSIVE-HINT PROMPTING (Zheng et al., 2023)
# ============================================================================

def progressive_hint(question, model="gemma3:1b", question_type="multiple_choice"):
    """
    Progressively add hints if model is uncertain
    """
    
    # Try without hints first
    answer1 = ask_ollama(f"{question}\n\nAnswer:", model=model, temperature=0, num_predict=30)
    extracted1 = extract_answer(answer1, question_type=question_type)
    
    # Add general hint
    hint_prompt = f"""{question}

Hint: Think about the most common or typical scenario.

Answer:"""
    
    answer2 = ask_ollama(hint_prompt, model=model, temperature=0, num_predict=30)
    extracted2 = extract_answer(answer2, question_type=question_type)
    
    # Vote
    if extracted1 == extracted2:
        return extracted1
    else:
        # Add stronger hint
        strong_hint = f"""{question}

Hint: Eliminate clearly impossible options first, then choose.

Answer:"""
        answer3 = ask_ollama(strong_hint, model=model, temperature=0, num_predict=30)
        return extract_answer(answer3, question_type=question_type)


# ============================================================================
# 8. MULTI-PERSONA PROMPTING (2024)
# ============================================================================

def multi_persona(question, model="gemma3:1b", question_type="multiple_choice"):
    """
    Get answers from multiple "personas", then vote
    """
    
    personas = [
        "You are a logical thinker who analyzes systematically.",
        "You are a creative thinker who uses intuition and experience.",
        "You are a critical thinker who questions assumptions."
    ]
    
    answers = []
    
    for persona in personas:
        prompt = f"""{persona}

{question}

Answer with just the letter:"""
        
        try:
            response = ask_ollama(prompt, model=model, temperature=0.3, num_predict=30)
            answer = extract_answer(response, question_type=question_type)
            if answer:
                answers.append(answer)
        except:
            continue
    
    if answers:
        return Counter(answers).most_common(1)[0][0]
    return ""


# ============================================================================
# 9. ENSEMBLE OF SOTA METHODS
# ============================================================================

def sota_ensemble(question, model="gemma3:1b", question_type="multiple_choice"):
    """
    Combine multiple SOTA methods with weighted voting
    """
    results = []
    
    # Method 1: Plan-and-Solve (weight: 2)
    try:
        ans = plan_and_solve(question, model, question_type)
        if ans:
            results.extend([ans, ans])
    except:
        pass
    
    # Method 2: Tree-of-Thoughts Light (weight: 2)
    try:
        ans = tree_of_thoughts_light(question, model, question_type, branches=3)
        if ans:
            results.extend([ans, ans])
    except:
        pass
    
    # Method 3: Progressive Hint (weight: 1)
    try:
        ans = progressive_hint(question, model, question_type)
        if ans:
            results.append(ans)
    except:
        pass
    
    # Method 4: Multi-Persona (weight: 1)
    try:
        ans = multi_persona(question, model, question_type)
        if ans:
            results.append(ans)
    except:
        pass
    
    # Vote
    if results:
        return Counter(results).most_common(1)[0][0]
    return ""