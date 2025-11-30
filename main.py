# main.py - Final Production Version
from benchmark import evaluate
from data_utils import load_gsm8k, load_strategyqa, load_commonsenseqa
import time
import os

MODEL = "gemma3:1b"

# ============================================================================
# CONFIGURATION: Choose your evaluation mode
# ============================================================================

# MODE 1: FULL EVALUATION (All data - will take several hours!)
# DATASET_SIZES = {
#     'gsm8k': None,          # ~1,319 samples
#     'strategyqa': None,     # ~2,000+ samples  
#     'commonsenseqa': None   # ~1,221 samples
# }

# MODE 2: QUICK TEST (Uncomment to use)
# DATASET_SIZES = {
#     'gsm8k': 50,
#     'strategyqa': 50,
#     'commonsenseqa': 50
# }

# MODE 3: SAMPLE TEST (Uncomment to use)
DATASET_SIZES = {
    'gsm8k': 100,
    'strategyqa': 100,
    'commonsenseqa': 100
}

# ============================================================================
# METHOD SELECTION
# ============================================================================

# OPTION 1: RECOMMENDED - Use "final-best" (your empirical results)
# METHODS = ["final-best"]

# OPTION 2: NEW - Use "hybrid-best" (combines L2M + PoT for GSM8K)
# This uses ultimate-gsm8k for math, which combines:
#   - Least-to-Most decomposition
#   - Program-of-Thought code generation
#   - Self-Consistency voting
# METHODS = ["hybrid-best"]

# OPTION 3: Compare baseline vs final vs hybrid
METHODS = ["baseline", "hybrid-best"]

# OPTION 4: Test specific hybrid methods for GSM8K
# METHODS = ["least-to-most", "pot-sc", "l2m-pot", "l2m-pot-sc", "ultimate-gsm8k"]

# OPTION 5: Full comparison (will take a LONG time!)
# METHODS = ["baseline", "cot", "pot-sc", "least-to-most", "l2m-pot-sc", "final-best", "hybrid-best"]

# ============================================================================

def estimate_time(datasets, methods):
    """Estimate total runtime based on empirical data"""
    total_samples = sum(len(ds[1]) for ds in datasets)
    
    # Empirical time estimates (seconds per sample) from your run
    time_per_sample = {
        'baseline': 2,
        'final-best': 35,  # Average: (5 + 15 + 85) / 3 ≈ 35
        'least-to-most': 5,
        'cot': 15,
        'powerful-sc': 85,  # ~80s based on your CommonsenseQA run
        'sota-ensemble': 85,
        'self-consistency': 20,
        'pot': 8,
        'pot-sc': 25
    }
    
    total_seconds = 0
    for method in methods:
        total_seconds += total_samples * time_per_sample.get(method, 10)
    
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    
    return hours, minutes, total_seconds


def print_header():
    """Print evaluation header"""
    print("="*70)
    print("LLM REASONING BENCHMARK - FINAL EVALUATION")
    print("="*70)
    print(f"Model: {MODEL}")
    print()
    
    # Show configuration
    if all(v is None for v in DATASET_SIZES.values()):
        print("⚠️  MODE: FULL EVALUATION")
        print("   Using ALL available data - this will take HOURS!")
    else:
        print("MODE: Sample Evaluation")
        for ds, size in DATASET_SIZES.items():
            if size:
                print(f"  - {ds}: {size} samples")
    print()


def print_method_info():
    """Print information about selected methods"""
    print("Selected Methods:")
    for method in METHODS:
        if method == "final-best":
            print(f"  - {method}: Empirically optimal methods")
            print(f"    • GSM8K (math): least-to-most")
            print(f"    • StrategyQA (boolean): chain-of-thought")
            print(f"    • CommonsenseQA (MC): powerful-sc")
        else:
            print(f"  - {method}")
    print()


def confirm_large_run(total_samples, hours, minutes):
    """Ask for confirmation on large evaluations"""
    if total_samples > 100:
        print("⚠️  LARGE EVALUATION DETECTED")
        print(f"   Total samples: {total_samples}")
        print(f"   Estimated time: ", end="")
        if hours > 0:
            print(f"{hours}h {minutes}m")
        else:
            print(f"{minutes}m")
        print()
        print("   Checkpoints will be saved every 5 samples.")
        print("   You can safely stop and resume anytime.")
        print()
        
        response = input("Continue with this evaluation? (y/n): ")
        if response.lower() != 'y':
            print("\n✗ Evaluation cancelled.")
            return False
        print()
    return True


def check_existing_results(datasets, methods):
    """Check for existing result files"""
    results_dir = "results"
    if not os.path.exists(results_dir):
        return
    
    existing = []
    for name, dataset, _ in datasets:
        for method in methods:
            csv_file = os.path.join(results_dir, f"{name}_{method}.csv")
            checkpoint_file = os.path.join(results_dir, f"{name}_{method}_checkpoint.json")
            
            if os.path.exists(csv_file):
                existing.append(f"{name}_{method} (complete)")
            elif os.path.exists(checkpoint_file):
                existing.append(f"{name}_{method} (partial)")
    
    if existing:
        print("Existing results found:")
        for item in existing:
            print(f"  ✓ {item}")
        print("\nThese will be resumed or skipped as appropriate.\n")


def main():
    print_header()
    print_method_info()
    
    # Load datasets
    print("Loading datasets...")
    gsm8k = load_gsm8k(DATASET_SIZES['gsm8k'])
    strategyqa = load_strategyqa(DATASET_SIZES['strategyqa'])
    commonsenseqa = load_commonsenseqa(DATASET_SIZES['commonsenseqa'])
    
    print(f"  ✓ GSM8K: {len(gsm8k)} samples")
    print(f"  ✓ StrategyQA: {len(strategyqa)} samples")
    print(f"  ✓ CommonsenseQA: {len(commonsenseqa)} samples")
    print()
    
    datasets = [
        ("gsm8k", gsm8k, "math"),
        ("strategyqa", strategyqa, "boolean"),
        ("commonsenseqa", commonsenseqa, "multiple_choice")
    ]
    
    # Check for existing results
    check_existing_results(datasets, METHODS)
    
    # Estimate time
    hours, minutes, total_seconds = estimate_time(datasets, METHODS)
    total_samples = sum(len(ds[1]) for ds in datasets)
    total_queries = total_samples * len(METHODS)
    
    print(f"Evaluation Plan:")
    print(f"  • Total samples: {total_samples}")
    print(f"  • Total queries: {total_queries}")
    print(f"  • Methods: {len(METHODS)}")
    print(f"  • Estimated time: ", end="")
    if hours > 0:
        print(f"{hours}h {minutes}m")
    else:
        print(f"{minutes}m")
    print()
    
    # Confirm for large runs
    if not confirm_large_run(total_samples, hours, minutes):
        return
    
    # Run evaluation
    results_summary = {}
    start_time = time.time()
    
    for method_idx, method in enumerate(METHODS, 1):
        print(f"{'#'*70}")
        print(f"METHOD {method_idx}/{len(METHODS)}: {method.upper()}")
        print(f"{'#'*70}\n")
        
        for ds_idx, (name, dataset, question_type) in enumerate(datasets, 1):
            if len(dataset) == 0:
                print(f"  ⚠ {name} is empty, skipping...\n")
                continue
            
            print(f"Dataset {ds_idx}/{len(datasets)}: {name.upper()}")
            print(f"Question type: {question_type}")
            print()
            
            acc = evaluate(
                dataset=dataset,
                name=name,
                method=method,
                model=MODEL,
                temperature=0.7,
                question_type=question_type,
                resume=True,  # Enable checkpoint resume
                checkpoint_interval=5  # Save every 5 samples
            )
            
            results_summary[f"{name}_{method}"] = acc
            
            # Show progress
            elapsed = time.time() - start_time
            completed = (method_idx - 1) * len(datasets) + ds_idx
            total_tasks = len(METHODS) * len(datasets)
            
            if completed < total_tasks:
                avg_time_per_task = elapsed / completed
                remaining_tasks = total_tasks - completed
                eta_seconds = avg_time_per_task * remaining_tasks
                eta_hours = int(eta_seconds // 3600)
                eta_minutes = int((eta_seconds % 3600) // 60)
                
                print(f"Progress: {completed}/{total_tasks} tasks completed")
                if eta_hours > 0:
                    print(f"ETA: ~{eta_hours}h {eta_minutes}m remaining\n")
                else:
                    print(f"ETA: ~{eta_minutes}m remaining\n")
    
    # Final Summary
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    
    for dataset_name in ["gsm8k", "strategyqa", "commonsenseqa"]:
        print(f"\n{dataset_name.upper()}:")
        for method in METHODS:
            key = f"{dataset_name}_{method}"
            if key in results_summary:
                acc = results_summary[key]
                print(f"  {method:20s}: {acc:6.2%}")
    
    print("\n" + "="*70)
    
    if results_summary:
        overall_acc = sum(results_summary.values()) / len(results_summary)
        print(f"\nOverall Accuracy: {overall_acc:.2%}")
        
        # Time taken
        total_time = time.time() - start_time
        time_hours = int(total_time // 3600)
        time_minutes = int((total_time % 3600) // 60)
        print(f"Total Time: ", end="")
        if time_hours > 0:
            print(f"{time_hours}h {time_minutes}m")
        else:
            print(f"{time_minutes}m")
        
        # Performance assessment
        print("\n" + "="*70)
        print("ASSESSMENT")
        print("="*70)
        
        if overall_acc < 0.20:
            print("\n⚠ Very low accuracy detected")
            print("  Recommendations:")
            print("  • Run: python check_results.py")
            print("  • Model may be too small for these tasks")
            print("  • Consider: gemma3:4b or llama3.2:3b")
        elif overall_acc < 0.40:
            print("\n✓ Normal performance for small models")
            print("  • This is expected for gemma3:1b on complex reasoning")
            print("  • For better results, try larger models:")
            print("    - gemma3:4b")
            print("    - llama3.2:3b")
            print("    - qwen2.5:7b")
        else:
            print("\n✓ Excellent results for this model size!")
            print("  • Performance exceeds expectations")
        
        # Save summary
        summary_file = "results/evaluation_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("EVALUATION SUMMARY\n")
            f.write("="*70 + "\n\n")
            f.write(f"Model: {MODEL}\n")
            f.write(f"Total Time: {time_hours}h {time_minutes}m\n")
            f.write(f"Overall Accuracy: {overall_acc:.2%}\n\n")
            
            for dataset_name in ["gsm8k", "strategyqa", "commonsenseqa"]:
                f.write(f"\n{dataset_name.upper()}:\n")
                for method in METHODS:
                    key = f"{dataset_name}_{method}"
                    if key in results_summary:
                        acc = results_summary[key]
                        f.write(f"  {method:20s}: {acc:6.2%}\n")
        
        print(f"\n✓ Summary saved to: {summary_file}")
    
    print("="*70)
    print("\nEvaluation complete! 🎉")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Evaluation interrupted by user")
        print("✓ Progress has been saved in checkpoint files")
        print("✓ Run again to resume from where you left off")
    except Exception as e:
        print(f"\n\n✗ Unexpected error: {e}")
        print("✓ Progress has been saved in checkpoint files")
        print("✓ Run again to resume from where you left off")