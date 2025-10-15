import re

def normalize_answer(ans):
    """
    Normalize the answer for comparison.
    Removes punctuation, spaces, and lowercases text.
    """
    ans = str(ans).lower()
    ans = re.sub(r"[^a-z0-9]", "", ans)
    return ans
