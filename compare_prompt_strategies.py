"""
Enhanced comparison of different prompt strategies with detailed analysis
"""

from ollama_utils import ask_ollama, extract_answer, format_prompt
from data_utils import load_strategyqa, load_commonsenseqa
import time

def test_strategy(question, expected, question_type, strategy_name, prompt_fn):
    """Test a single strategy and return results"""
    try:
        prompt = prompt_fn()
        response = ask_ollama(prompt, model="gemma3:1b", temperature=0)
        extracted = extract_answer(response, question_type=question_type)
        is_correct = extracted.lower() == expected.lower()
        return {
            'response': response,
            'extracted': extracted,
            'correct': is_correct,
            'prompt_length': len(prompt)
        }
    except Exception as e:
        return {
            'response': f"ERROR: {e}",
            'extracted': "",
            'correct': False,
            'prompt_length': 0
        }


def test_prompt_strategies():
    """Enhanced comparison with statistics"""
    
    print("="*80)
    print("ENHANCED PROMPT STRATEGY COMPARISON")
    print("="*80)
    print("\nThis will test multiple strategies and show which works best.\n")
    
    # Statistics tracking
    stats = {
        'strategyqa': {'strategy1': [], 'strategy2': [], 'strategy3': [], 'strategy4': []},
        'commonsenseqa': {'strategy1': [], 'strategy2': [], 'strategy3': [], 'strategy4': []}
    }
    
    # ============ STRATEGYQA ============
    print("\n" + "▓"*80)
    print("PART 1: STRATEGYQA (Yes/No Questions)")
    print("▓"*80 + "\n")
    
    strategyqa = load_strategyqa(3)  # Test 3 questions
    
    for i, item in enumerate(strategyqa, 1):
        question = item['question']
        expected = item['answer']
        
        print(f"\n{'='*80}")
        print(f"Question {i}: {question}")
        print(f"Expected Answer: '{expected}'")
        print(f"{'='*80}\n")
        
        # Strategy 1: Direct instruction
        print("┌─ [Strategy 1: Direct instruction, no examples]")
        result1 = test_strategy(
            question, expected, "boolean",
            "Direct",
            lambda: format_prompt(question, question_type="boolean", use_few_shot=False)
        )
        print(f"│  Prompt length: {result1['prompt_length']} chars")
        print(f"│  Model response: '{result1['response']}'")
        print(f"│  Extracted: '{result1['extracted']}'")
        print(f"└─ Result: {'✓ CORRECT' if result1['correct'] else '✗ WRONG'}\n")
        stats['strategyqa']['strategy1'].append(result1['correct'])
        time.sleep(0.5)
        
        # Strategy 2: Few-shot
        print("┌─ [Strategy 2: Few-shot examples]")
        result2 = test_strategy(
            question, expected, "boolean",
            "Few-shot",
            lambda: format_prompt(question, question_type="boolean", use_few_shot=True)
        )
        print(f"│  Prompt length: {result2['prompt_length']} chars")
        print(f"│  Model response: '{result2['response']}'")
        print(f"│  Extracted: '{result2['extracted']}'")
        print(f"└─ Result: {'✓ CORRECT' if result2['correct'] else '✗ WRONG'}\n")
        stats['strategyqa']['strategy2'].append(result2['correct'])
        time.sleep(0.5)
        
        # Strategy 3: Ultra-minimal
        print("┌─ [Strategy 3: Ultra-minimal]")
        result3 = test_strategy(
            question, expected, "boolean",
            "Minimal",
            lambda: f"{question}\n\nAnswer (yes or no):"
        )
        print(f"│  Prompt length: {result3['prompt_length']} chars")
        print(f"│  Model response: '{result3['response']}'")
        print(f"│  Extracted: '{result3['extracted']}'")
        print(f"└─ Result: {'✓ CORRECT' if result3['correct'] else '✗ WRONG'}\n")
        stats['strategyqa']['strategy3'].append(result3['correct'])
        time.sleep(0.5)
        
        # Strategy 4: Chain of thought
        print("┌─ [Strategy 4: Explicit reasoning]")
        result4 = test_strategy(
            question, expected, "boolean",
            "CoT",
            lambda: f"{question}\n\nThink step by step, then answer yes or no.\nReasoning:"
        )
        print(f"│  Prompt length: {result4['prompt_length']} chars")
        print(f"│  Model response: '{result4['response']}'")
        print(f"│  Extracted: '{result4['extracted']}'")
        print(f"└─ Result: {'✓ CORRECT' if result4['correct'] else '✗ WRONG'}\n")
        stats['strategyqa']['strategy4'].append(result4['correct'])
        time.sleep(0.5)
        
        print(f"{'─'*80}\n")
    
    # ============ COMMONSENSEQA ============
    print("\n" + "▓"*80)
    print("PART 2: COMMONSENSEQA (Multiple Choice)")
    print("▓"*80 + "\n")
    
    commonsenseqa = load_commonsenseqa(3)  # Test 3 questions
    
    for i, item in enumerate(commonsenseqa, 1):
        question = item['question']
        expected = item['answer']
        
        print(f"\n{'='*80}")
        print(f"Question {i}: {question[:120]}...")
        print(f"Expected Answer: '{expected}'")
        print(f"{'='*80}\n")
        
        # Strategy 1: Direct instruction
        print("┌─ [Strategy 1: Direct instruction, no examples]")
        result1 = test_strategy(
            question, expected, "multiple_choice",
            "Direct",
            lambda: format_prompt(question, question_type="multiple_choice", use_few_shot=False)
        )
        print(f"│  Prompt length: {result1['prompt_length']} chars")
        print(f"│  Model response: '{result1['response']}'")
        print(f"│  Extracted: '{result1['extracted']}'")
        print(f"└─ Result: {'✓ CORRECT' if result1['correct'] else '✗ WRONG'}\n")
        stats['commonsenseqa']['strategy1'].append(result1['correct'])
        time.sleep(0.5)
        
        # Strategy 2: Few-shot
        print("┌─ [Strategy 2: Few-shot examples]")
        result2 = test_strategy(
            question, expected, "multiple_choice",
            "Few-shot",
            lambda: format_prompt(question, question_type="multiple_choice", use_few_shot=True)
        )
        print(f"│  Prompt length: {result2['prompt_length']} chars")
        print(f"│  Model response: '{result2['response']}'")
        print(f"│  Extracted: '{result2['extracted']}'")
        print(f"└─ Result: {'✓ CORRECT' if result2['correct'] else '✗ WRONG'}\n")
        stats['commonsenseqa']['strategy2'].append(result2['correct'])
        time.sleep(0.5)
        
        # Strategy 3: Ultra-minimal
        print("┌─ [Strategy 3: Ultra-minimal]")
        result3 = test_strategy(
            question, expected, "multiple_choice",
            "Minimal",
            lambda: f"{question}\n\nBest answer:"
        )
        print(f"│  Prompt length: {result3['prompt_length']} chars")
        print(f"│  Model response: '{result3['response']}'")
        print(f"│  Extracted: '{result3['extracted']}'")
        print(f"└─ Result: {'✓ CORRECT' if result3['correct'] else '✗ WRONG'}\n")
        stats['commonsenseqa']['strategy3'].append(result3['correct'])
        time.sleep(0.5)
        
        # Strategy 4: Reasoning first
        print("┌─ [Strategy 4: Reason then choose]")
        result4 = test_strategy(
            question, expected, "multiple_choice",
            "Reason",
            lambda: f"{question}\n\nAnalyze each option, then state your answer as a single letter:"
        )
        print(f"│  Prompt length: {result4['prompt_length']} chars")
        print(f"│  Model response: '{result4['response']}'")
        print(f"│  Extracted: '{result4['extracted']}'")
        print(f"└─ Result: {'✓ CORRECT' if result4['correct'] else '✗ WRONG'}\n")
        stats['commonsenseqa']['strategy4'].append(result4['correct'])
        time.sleep(0.5)
        
        print(f"{'─'*80}\n")
    
    # ============ RESULTS SUMMARY ============
    print("\n" + "█"*80)
    print("RESULTS SUMMARY")
    print("█"*80 + "\n")
    
    # StrategyQA summary
    print("╔" + "═"*78 + "╗")
    print("║ STRATEGYQA (Yes/No) - Accuracy by Strategy                              ║")
    print("╠" + "═"*78 + "╣")
    
    for strategy_num in range(1, 5):
        strategy_name = ['Direct', 'Few-shot', 'Minimal', 'Reasoning'][strategy_num-1]
        results = stats['strategyqa'][f'strategy{strategy_num}']
        correct = sum(results)
        total = len(results)
        accuracy = (correct / total * 100) if total > 0 else 0
        
        bar_length = int(accuracy / 2)  # Scale to 50 chars
        bar = "█" * bar_length + "░" * (50 - bar_length)
        
        print(f"║ Strategy {strategy_num} ({strategy_name:12s}): {correct}/{total} = {accuracy:5.1f}% │{bar}│ ║")
    
    print("╚" + "═"*78 + "╝\n")
    
    # CommonsenseQA summary
    print("╔" + "═"*78 + "╗")
    print("║ COMMONSENSEQA (Multiple Choice) - Accuracy by Strategy                  ║")
    print("╠" + "═"*78 + "╣")
    
    for strategy_num in range(1, 5):
        strategy_name = ['Direct', 'Few-shot', 'Minimal', 'Reasoning'][strategy_num-1]
        results = stats['commonsenseqa'][f'strategy{strategy_num}']
        correct = sum(results)
        total = len(results)
        accuracy = (correct / total * 100) if total > 0 else 0
        
        bar_length = int(accuracy / 2)
        bar = "█" * bar_length + "░" * (50 - bar_length)
        
        print(f"║ Strategy {strategy_num} ({strategy_name:12s}): {correct}/{total} = {accuracy:5.1f}% │{bar}│ ║")
    
    print("╚" + "═"*78 + "╝\n")
    
    # ============ RECOMMENDATIONS ============
    print("╔" + "═"*78 + "╗")
    print("║ RECOMMENDATIONS                                                          ║")
    print("╚" + "═"*78 + "╝\n")
    
    # Find best strategy for each type
    sq_scores = {f'Strategy {i+1}': sum(stats['strategyqa'][f'strategy{i+1}']) 
                 for i in range(4)}
    csqa_scores = {f'Strategy {i+1}': sum(stats['commonsenseqa'][f'strategy{i+1}']) 
                   for i in range(4)}
    
    best_sq = max(sq_scores.items(), key=lambda x: x[1])
    best_csqa = max(csqa_scores.items(), key=lambda x: x[1])
    
    print(f"✓ Best for StrategyQA: {best_sq[0]} ({best_sq[1]}/{len(stats['strategyqa']['strategy1'])} correct)")
    print(f"✓ Best for CommonsenseQA: {best_csqa[0]} ({best_csqa[1]}/{len(stats['commonsenseqa']['strategy1'])} correct)\n")
    
    # Overall assessment
    overall_accuracy = (sum(sq_scores.values()) + sum(csqa_scores.values())) / (
        len(stats['strategyqa']['strategy1']) * 4 + len(stats['commonsenseqa']['strategy1']) * 4
    ) * 100
    
    print(f"Overall model accuracy: {overall_accuracy:.1f}%\n")
    
    if overall_accuracy < 30:
        print("⚠ WARNING: Accuracy is very low (<30%)")
        print("  → gemma3:1b is too small for these tasks")
        print("  → RECOMMENDED: Upgrade to gemma3:4b or llama3.2:3b")
        print("  → Command: ollama pull gemma3:4b")
    elif overall_accuracy < 50:
        print("⚠ NOTICE: Accuracy is moderate (30-50%)")
        print("  → gemma3:1b can handle some cases but struggles")
        print("  → Consider upgrading to gemma3:4b for better results")
    else:
        print("✓ Good accuracy! Model is performing well.")
        print("  → Continue using gemma3:1b or try gemma3:4b for even better results")
    
    print("\n" + "="*80)
    print("Update benchmark.py to use the best-performing strategy!")
    print("="*80 + "\n")


if __name__ == "__main__":
    print("This will test 4 different prompt strategies on 3 questions each (24 queries total).")
    print("Make sure Ollama is running (ollama serve)")
    print("\nEstimated time: 2-3 minutes\n")
    input("Press Enter to continue...")
    
    test_prompt_strategies()