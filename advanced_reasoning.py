"""
Advanced reasoning methods for CommonsenseQA
"""

from ollama_utils import ask_ollama, extract_answer
from collections import Counter
import re

def analyze_each_option(question, model="gemma3:1b"):
    """
    Analyze each option separately and score them
    Better than asking model to choose directly
    """
    # Parse question and options
    main_q = question.split('(A)')[0].strip()
    
    # Extract options
    options = {}
    for letter in ['A', 'B', 'C', 'D', 'E']:
        pattern = f'\({letter}\)\s*([^(]+?)(?=\([A-E]\)|$)'
        match = re.search(pattern, question)
        if match:
            options[letter] = match.group(1).strip()
    
    if not options:
        return None
    
    # Score each option
    scores = {}
    for letter, option_text in options.items():
        prompt = f"""Question: {main_q}
Option {letter}: {option_text}

Does this option make sense as an answer? Rate 1-10.
Score:"""
        
        try:
            response = ask_ollama(prompt, model=model, temperature=0, num_predict=10)
            # Extract number
            numbers = re.findall(r'\d+', response)
            if numbers:
                scores[letter] = int(numbers[0])
            else:
                scores[letter] = 5  # neutral
        except:
            scores[letter] = 5
    
    # Return highest scored option
    if scores:
        return max(scores.items(), key=lambda x: x[1])[0]
    return None


def contrastive_prompting(question, model="gemma3:1b"):
    """
    Ask "Why NOT option X?" to eliminate wrong answers
    """
    # Parse options
    options = []
    for letter in ['A', 'B', 'C', 'D', 'E']:
        pattern = f'\({letter}\)'
        if pattern in question:
            options.append(letter)
    
    if len(options) < 2:
        return None
    
    # Ask why NOT each option
    elimination_votes = {}
    
    for letter in options:
        prompt = f"""{question}

Why is option {letter} NOT the best answer? Answer briefly:"""
        
        try:
            response = ask_ollama(prompt, model=model, temperature=0.3, num_predict=50)
            
            # Check if response says it's actually good
            positive_words = ['is good', 'is correct', 'makes sense', 'is the answer']
            if any(word in response.lower() for word in positive_words):
                elimination_votes[letter] = -1  # This is likely the answer
            else:
                elimination_votes[letter] = 1  # Eliminated
        except:
            elimination_votes[letter] = 0
    
    # Find option with lowest elimination score (least eliminated = best)
    if elimination_votes:
        return min(elimination_votes.items(), key=lambda x: x[1])[0]
    return None


def chain_of_thought_voting(question, model="gemma3:1b", samples=5):
    """
    Improved self-consistency with better CoT prompts for CommonsenseQA
    """
    prompts_variants = [
        # Variant 1: Think about context
        f"""{question}

Think about the context and real-world knowledge. Which option makes most sense?
Answer with just the letter:""",
        
        # Variant 2: Eliminate wrong answers
        f"""{question}

First eliminate clearly wrong options, then choose from remaining.
Answer:""",
        
        # Variant 3: Common sense reasoning
        f"""{question}

Use common sense and everyday experience. What would most people choose?
Answer:""",
        
        # Variant 4: Direct
        f"""{question}

Answer with the letter of the best option:""",
        
        # Variant 5: Step by step
        f"""{question}

Think step by step about each option. Then answer with just the letter:"""
    ]
    
    answers = []
    
    for i in range(samples):
        prompt = prompts_variants[i % len(prompts_variants)]
        
        try:
            response = ask_ollama(prompt, model=model, temperature=0.5, num_predict=30)
            answer = extract_answer(response, question_type="multiple_choice")
            if answer:
                answers.append(answer)
        except:
            continue
    
    if not answers:
        return None
    
    # Majority vote
    answer_counts = Counter(answers)
    return answer_counts.most_common(1)[0][0]


def ensemble_commonsense(question, model="gemma3:1b"):
    """
    Ensemble of multiple strategies for CommonsenseQA
    Combines: analyze_each, contrastive, and CoT voting
    """
    results = []
    
    # Method 1: Analyze each option (weight: 2)
    try:
        ans1 = analyze_each_option(question, model=model)
        if ans1:
            results.extend([ans1, ans1])  # Add twice for higher weight
    except:
        pass
    
    # Method 2: Contrastive (weight: 1)
    try:
        ans2 = contrastive_prompting(question, model=model)
        if ans2:
            results.append(ans2)
    except:
        pass
    
    # Method 3: CoT voting (weight: 2)
    try:
        ans3 = chain_of_thought_voting(question, model=model, samples=3)
        if ans3:
            results.extend([ans3, ans3])
    except:
        pass
    
    # Vote
    if results:
        answer_counts = Counter(results)
        return answer_counts.most_common(1)[0][0]
    
    return None


def smart_self_consistency_mc(question, model="gemma3:1b"):
    """
    Optimized self-consistency specifically for multiple choice
    Uses diverse prompting strategies
    """
    return chain_of_thought_voting(question, model=model, samples=5)