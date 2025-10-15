# from data_utils import load_gsm8k, load_strategyqa, load_commonsenseqa
# from benchmark import evaluate

# MODEL = "gemma3:1b"

# def main():
#     gsm8k = load_gsm8k(30)
#     stratqa = load_strategyqa(30)
#     csqa = load_commonsenseqa(30)

#     for method in ["baseline", "self-consistency", "program-of-thought", "grounded"]:
#         print(f"\n=== METHOD: {method} ===")
#         evaluate(gsm8k, "gsm8k", method=method, model=MODEL)
#         evaluate(stratqa, "strategyqa", method=method, model=MODEL)
#         evaluate(csqa, "commonsenseqa", method=method, model=MODEL)

# if __name__ == "__main__":
#     main()
from benchmark import evaluate
from data_utils import load_gsm8k, load_strategyqa, load_commonsenseqa

MODEL = "gemma3:1b"

def main():
    # Load datasets
    gsm8k = load_gsm8k(3)          # load 30 examples
    strategyqa = load_strategyqa(3)
    commonsenseqa = load_commonsenseqa(3)

    datasets = [
        ("GSM8K", gsm8k),
        ("STRATEGYQA", strategyqa),
        ("COMMONSENSEQA", commonsenseqa)
    ]

    methods = ["baseline", "self-consistency"]

    for method in methods:
        for name, dataset in datasets:
            evaluate(dataset, name.lower(), method=method, model=MODEL)

if __name__ == "__main__":
    main()
