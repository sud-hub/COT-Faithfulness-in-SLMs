from datasets import load_dataset
import os

def load_gsm8k(n=50):
    data = load_dataset("openai/gsm8k", "main", split=f"test[:{n}]")
    return [{"question": ex["question"], "answer": ex["answer"]} for ex in data]

def load_commonsenseqa(n=50):
    data = load_dataset("tau/commonsense_qa", split=f"validation[:{n}]")
    out = []
    for ex in data:
        q = ex.get("question") or ex.get("question__stem")
        ans = ex.get("answerKey") or ex.get("label") or ex.get("answer")
        if isinstance(ans, int):
            ans = str(ans)
        out.append({"question": q, "answer": str(ans).strip()})
    return out

def load_strategyqa(n=50):
    path = "data/strategyqa_train_cleaned.json"
    if not os.path.exists(path):
        raise FileNotFoundError(
            "StrategyQA cleaned JSON not found. Please run preprocess_strategyqa.py first."
        )
    data = load_dataset("json", data_files=path, split=f"train[:{n}]")
    out = []
    for ex in data:
        q = ex.get("question") or "N/A"
        a = ex.get("answer") or "yes"
        out.append({"question": q, "answer": str(a).strip()})
    return out
