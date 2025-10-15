import json

for split in ["train", "test"]:
    with open(f"data/strategyqa_{split}.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    cleaned = []
    for ex in data:
        ex["evidence"] = str(ex.get("evidence", ""))
        # Ensure question and answer fields exist
        if "question" not in ex:
            ex["question"] = ex.get("Question") or ex.get("question_text") or "N/A"
        if "answer" not in ex:
            a = ex.get("label") or ex.get("final_answer") or "yes"
            if isinstance(a, bool):
                a = "yes" if a else "no"
            elif isinstance(a, list):
                a = a[0] if a else "yes"
            ex["answer"] = str(a)
        cleaned.append(ex)
    with open(f"data/strategyqa_{split}_cleaned.json", "w", encoding="utf-8") as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=2)
    print(f"{split} cleaned, {len(cleaned)} examples")
