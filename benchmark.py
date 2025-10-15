# import re, csv, time
# from tqdm import tqdm
# from reasoning_methods import self_consistency, grounded_prompt, program_of_thought
# from ollama_utils import ask_ollama

# def normalize_answer(ans):
#     ans = ans.lower().strip()
#     ans = re.sub(r'[^0-9a-z\s]', '', ans)
#     return ans

# def evaluate(dataset, task_name, method="baseline", model="gemma3:1b"):
#     results = []
#     correct = 0
#     start = time.time()

#     for sample in tqdm(dataset, desc=f"{task_name.upper()} [{method}]"):
#         q, gold = sample["question"], sample["answer"]

#         if method == "baseline":
#             prompt = f"Question: {q}\nAnswer with only the final answer."
#             pred = ask_ollama(prompt, model=model)
#         elif method == "self-consistency":
#             prompt = f"Question: {q}\nAnswer with only the final number or 'yes/no'."
#             pred = self_consistency(prompt, model=model)
#         elif method == "program-of-thought":
#             pred = str(program_of_thought(q, model=model))
#         elif method == "grounded":
#             context = "General world knowledge about common objects and relations."
#             prompt = grounded_prompt(q, context)
#             pred = ask_ollama(prompt, model=model)
#         else:
#             raise ValueError("Unknown method type!")

#         ok = normalize_answer(pred).startswith(normalize_answer(gold))
#         correct += int(ok)
#         results.append({"question": q, "pred": pred, "gold": gold, "correct": ok})

#     acc = correct / len(dataset)
#     runtime = time.time() - start
#     print(f"{task_name.upper()} | {method} | Accuracy: {acc:.2f} | Time: {runtime:.1f}s")

#     # Save CSV
#     csv_name = f"results/{task_name}_{method}.csv"
#     with open(csv_name, "w", newline="") as f:
#         writer = csv.DictWriter(f, fieldnames=results[0].keys())
#         writer.writeheader()
#         writer.writerows(results)

#     return acc, runtime
import os
import csv
from tqdm import tqdm
from ollama_utils import ask_ollama
from reasoning_methods import self_consistency

def evaluate(dataset, name, method="baseline", model="gemma3:1b", temperature=0.7):
    """
    Evaluate a dataset using baseline or self-consistency.
    Writes predictions to CSV.
    """
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    csv_name = os.path.join(results_dir, f"{name}_{method}.csv")

    results = []
    for i, item in enumerate(tqdm(dataset, desc=f"{name} [{method}]")):
        prompt = item['question']

        try:
            if method == "baseline":
                pred = ask_ollama(prompt, model=model)
            elif method == "self-consistency":
                pred = self_consistency(prompt, model=model, temperature=temperature)
            else:
                raise ValueError(f"Unknown method: {method}")
        except Exception as e:
            print(f"Error on sample {i}: {e}")
            pred = ""

        results.append({'question': prompt, 'prediction': pred})

    # Write CSV safely
    with open(csv_name, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["question", "prediction"])
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    # Placeholder accuracy
    correct = sum(1 for i, item in enumerate(dataset) if item.get("answer") == results[i]["prediction"])
    acc = correct / len(dataset) if len(dataset) > 0 else 0

    print(f"{name} | {method} | Accuracy: {acc:.2f} | Samples: {len(dataset)}")
