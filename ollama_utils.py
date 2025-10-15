import subprocess
import os

def ask_ollama(prompt, model="gemma3:1b", temperature=None, reasoning=None):
    """
    Sends a prompt to Ollama and returns the text output.
    Handles Windows UTF-8 decoding and optional parameters.
    """
    cmd = ["ollama", "run", model]

    # Optional flags
    if temperature is not None:
        cmd += ["--temperature", str(temperature)]
    if reasoning is not None:
        cmd += ["--reasoning", reasoning]

    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace"
        )
        out, err = proc.communicate(prompt)
        if err:
            print("Ollama error:", err.strip())
        return out.strip()
    except FileNotFoundError:
        raise RuntimeError("Ollama executable not found. Make sure Ollama is installed and in PATH.")
    except Exception as e:
        raise RuntimeError(f"Error running Ollama: {e}")
