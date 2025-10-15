from ollama_utils import ask_ollama

def self_consistency(prompt, model="gemma3:1b", temperature=0.7, samples=5):
    """
    Run self-consistency: multiple stochastic samples and majority vote
    """
    outputs = []
    for _ in range(samples):
        out = ask_ollama(prompt, model=model, temperature=temperature)
        outputs.append(out)

    # Majority vote
    vote = max(set(outputs), key=outputs.count)
    return vote
