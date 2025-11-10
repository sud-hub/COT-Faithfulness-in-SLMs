# benchmark.py - Final Optimized Version with Checkpointing
import os
import csv
import json
from tqdm import tqdm
from ollama_utils import ask_ollama, extract_answer, format_prompt
from reasoning_methods import (
    self_consistency,
    chain_of_thought,
    program_of_thought_method,
    pot_with_self_consistency
)
from advanced_reasoning import (
    smart_self_consistency_mc,
    ensemble_commonsense
)
from sota_reasoning import (
    least_to_most_prompting,
    self_refine,
    plan_and_solve,
    analogical_prompting,
    metacognitive_prompting,
    tree_of_thoughts_light,
    progressive_hint,
    multi_persona,
    sota_ensemble
)
from hybrid_reasoning import (
    least_to_most_pot,
    least_to_most_pot_sc,
    gsm8k_ensemble,
    ultimate_gsm8k
)

def normalize_answer(answer, question_type):
    """Normalize answers for comparison"""
    answer_str = str(answer).strip()
    if question_type == "math":
        import re
        normalized = re.sub(r'[$,\s]', '', answer_str)
        try:
            num = float(normalized)
            if num == int(num):
                return str(int(num))
            return str(num)
        except:
            return normalized
    elif question_type == "boolean":
        return answer_str.lower()
    elif question_type == "multiple_choice":
        letter_only = answer_str.split(':')[0].split(' ')[0].strip()
        return letter_only.upper()
    return answer_str


def load_checkpoint(checkpoint_file):
    """Load existing results from checkpoint"""
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                print(f"  ✓ Checkpoint found: {len(data['results'])}/{data['total']} samples completed")
                return data
        except Exception as e:
            print(f"  ⚠ Checkpoint corrupted, starting fresh: {e}")
            return None
    return None


def save_checkpoint(checkpoint_file, results, correct, total):
    """Save progress checkpoint"""
    data = {
        'results': results,
        'correct': correct,
        'total': total,
        'last_index': len(results) - 1
    }
    try:
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"  ⚠ Failed to save checkpoint: {e}")


def evaluate(dataset, name, method="baseline", model="gemma3:1b", temperature=0.7, 
             question_type="math", resume=True, checkpoint_interval=5):
    """
    Evaluate a dataset with checkpointing support
    
    Args:
        dataset: List of question/answer pairs
        name: Dataset name (e.g., 'gsm8k')
        method: Reasoning method to use
        model: LLM model name
        temperature: Sampling temperature
        question_type: Type of questions ('math', 'boolean', 'multiple_choice')
        resume: Whether to resume from checkpoint
        checkpoint_interval: Save checkpoint every N samples
    """
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    csv_name = os.path.join(results_dir, f"{name}_{method}.csv")
    checkpoint_file = os.path.join(results_dir, f"{name}_{method}_checkpoint.json")

    # --- Method mapping ---
    if method == "hybrid-best":
        # Use NEW hybrid methods for each benchmark
        lower_name = name.lower()
        if "gsm8k" in lower_name or question_type == "math":
            mapped_method = "ultimate-gsm8k"  # L2M-PoT-SC + Ensemble
        elif "strategyqa" in lower_name or question_type == "boolean":
            mapped_method = "cot"  # Keep best for boolean
        elif "commonsenseqa" in lower_name or question_type == "multiple_choice":
            mapped_method = "powerful-sc"  # Keep best for MC
        else:
            mapped_method = "baseline"
    elif method == "final-best":
        # Use empirically best method for each benchmark
        lower_name = name.lower()
        if "gsm8k" in lower_name or question_type == "math":
            mapped_method = "least-to-most"  # 5.00% (best for math)
        elif "strategyqa" in lower_name or question_type == "boolean":
            mapped_method = "cot"  # 57.50% (best for boolean)
        elif "commonsenseqa" in lower_name or question_type == "multiple_choice":
            mapped_method = "powerful-sc"  # 55.00% (best for MC)
        else:
            mapped_method = "baseline"
    elif method == "best":
        lower_name = name.lower()
        if "gsm8k" in lower_name or question_type == "math":
            mapped_method = "pot"
        elif "strategyqa" in lower_name or question_type == "boolean":
            mapped_method = "cot"
        elif "commonsenseqa" in lower_name or question_type == "multiple_choice":
            mapped_method = "powerful-sc"
        else:
            mapped_method = "baseline"
    elif method == "sota-best":
        lower_name = name.lower()
        if "gsm8k" in lower_name or question_type == "math":
            mapped_method = "least-to-most"
        elif "strategyqa" in lower_name or question_type == "boolean":
            mapped_method = "plan-and-solve"
        elif "commonsenseqa" in lower_name or question_type == "multiple_choice":
            mapped_method = "sota-ensemble"
        else:
            mapped_method = "baseline"
    else:
        mapped_method = method

    # Check for existing checkpoint
    results = []
    correct = 0
    start_idx = 0
    
    if resume:
        checkpoint_data = load_checkpoint(checkpoint_file)
        if checkpoint_data:
            results = checkpoint_data['results']
            correct = checkpoint_data['correct']
            start_idx = checkpoint_data['last_index'] + 1
            
            if start_idx >= len(dataset):
                print(f"  ✓ Already completed! Loading results...")
                acc = correct / len(dataset) if len(dataset) > 0 else 0
                return acc
            
            print(f"  → Resuming from sample {start_idx + 1}/{len(dataset)}")

    # Initialize progress bar
    pbar = tqdm(
        total=len(dataset), 
        initial=start_idx, 
        desc=f"{name} [{mapped_method}]",
        unit="sample"
    )

    # Process samples
    for i in range(start_idx, len(dataset)):
        item = dataset[i]
        prompt = item['question']
        ground_truth = str(item.get('answer', '')).strip()

        # Format prompt based on question type
        if question_type == "multiple_choice":
            formatted_prompt = format_prompt(prompt, question_type=question_type, 
                                            use_few_shot=False, use_reasoning=True)
        else:
            formatted_prompt = format_prompt(prompt, question_type=question_type, 
                                            use_few_shot=True, use_reasoning=False)

        try:
            # ============= APPLY REASONING METHOD =============
            if mapped_method == "baseline":
                raw_pred = ask_ollama(formatted_prompt, model=model, temperature=0)
                pred = extract_answer(raw_pred, question_type=question_type)

            elif mapped_method == "cot":
                pred = chain_of_thought(formatted_prompt, model=model, question_type=question_type)

            elif mapped_method == "self-consistency":
                pred = self_consistency(formatted_prompt, model=model, temperature=0.8, 
                                       samples=7, question_type=question_type)

            elif mapped_method == "pot":
                pred = program_of_thought_method(prompt, model=model)

            elif mapped_method == "pot-sc":
                pred = pot_with_self_consistency(prompt, model=model, samples=3)

            elif mapped_method == "powerful-sc":
                pred = smart_self_consistency_mc(prompt, model=model)

            elif mapped_method == "ensemble":
                pred = ensemble_commonsense(prompt, model=model)

            elif mapped_method == "least-to-most":
                pred = least_to_most_prompting(prompt, model=model, question_type=question_type)

            elif mapped_method == "self-refine":
                pred = self_refine(prompt, model=model, question_type=question_type, iterations=2)

            elif mapped_method == "plan-and-solve":
                pred = plan_and_solve(prompt, model=model, question_type=question_type)

            elif mapped_method == "analogical":
                pred = analogical_prompting(prompt, model=model, question_type=question_type)

            elif mapped_method == "metacognitive":
                pred = metacognitive_prompting(prompt, model=model, question_type=question_type)

            elif mapped_method == "tree-of-thoughts":
                pred = tree_of_thoughts_light(prompt, model=model, question_type=question_type, branches=3)

            elif mapped_method == "progressive-hint":
                pred = progressive_hint(prompt, model=model, question_type=question_type)

            elif mapped_method == "multi-persona":
                pred = multi_persona(prompt, model=model, question_type=question_type)

            elif mapped_method == "sota-ensemble":
                pred = sota_ensemble(prompt, model=model, question_type=question_type)

            # ============= HYBRID METHODS (L2M + PoT) =============
            elif mapped_method == "l2m-pot":
                pred = least_to_most_pot(prompt, model=model)

            elif mapped_method == "l2m-pot-sc":
                pred = least_to_most_pot_sc(prompt, model=model, samples=3)

            elif mapped_method == "gsm8k-ensemble":
                pred = gsm8k_ensemble(prompt, model=model)

            elif mapped_method == "ultimate-gsm8k":
                # Use fast_mode=True for 2x speed, fast_mode=False for max accuracy
                pred = ultimate_gsm8k(prompt, model=model, fast_mode=True)

            else:
                raise ValueError(f"Unknown method: {mapped_method}")

        except Exception as e:
            pbar.write(f"  ⚠ Error on sample {i + 1}: {e}")
            pred = ""

        # Normalize and compare
        normalized_pred = normalize_answer(pred, question_type)
        normalized_truth = normalize_answer(ground_truth, question_type)

        is_correct = False
        if question_type == "math":
            try:
                is_correct = abs(float(normalized_pred) - float(normalized_truth)) < 0.01
            except:
                is_correct = normalized_pred == normalized_truth
        else:
            is_correct = normalized_pred == normalized_truth

        if is_correct:
            correct += 1

        results.append({
            'question': prompt,
            'ground_truth': ground_truth,
            'prediction': pred,
            'correct': is_correct
        })
        
        # Update progress bar with current accuracy
        current_acc = correct / len(results) * 100
        pbar.set_postfix({'acc': f'{current_acc:.1f}%', 'correct': f'{correct}/{len(results)}'})
        pbar.update(1)
        
        # Save checkpoint periodically
        if (i + 1) % checkpoint_interval == 0:
            save_checkpoint(checkpoint_file, results, correct, len(dataset))

    pbar.close()

    # Final checkpoint save
    save_checkpoint(checkpoint_file, results, correct, len(dataset))

    # Write final CSV
    with open(csv_name, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["question", "ground_truth", "prediction", "correct"])
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    # Calculate and display results
    acc = correct / len(dataset) if len(dataset) > 0 else 0
    print(f"\n{'='*60}")
    print(f"Dataset: {name.upper()}")
    print(f"Method: {mapped_method}")
    print(f"Model: {model}")
    print(f"Accuracy: {acc:.2%} ({correct}/{len(dataset)})")
    print(f"Results saved to: {csv_name}")
    print(f"{'='*60}\n")
    
    # Clean up checkpoint after successful completion
    if os.path.exists(checkpoint_file):
        try:
            os.remove(checkpoint_file)
            print(f"  ✓ Checkpoint cleaned up\n")
        except:
            pass

    return acc