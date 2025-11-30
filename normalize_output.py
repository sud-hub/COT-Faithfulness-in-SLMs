import re

def extract_final_answer(output: str) -> str:
    """
    Extracts the final numeric/text answer from verbose model output.
    """
    if not output:
        return ""

    # LaTeX boxed answer
    boxed = re.search(r"\\boxed\{([^}]+)\}", output)
    if boxed:
        return boxed.group(1).strip()

    # Final Answer pattern
    final_answer = re.search(r"Final Answer:.*?([0-9\w\s\.-]+)", output, re.IGNORECASE)
    if final_answer:
        return final_answer.group(1).strip()

    # The answer is pattern
    answer_is = re.search(r"The answer is ([0-9\w\s\.-]+)", output, re.IGNORECASE)
    if answer_is:
        return answer_is.group(1).strip()

    # Fallback: last number
    last_number = re.findall(r"[-+]?\d*\.\d+|\d+", output)
    if last_number:
        return last_number[-1].strip()

    # Fallback: return full output
    return output.strip()
