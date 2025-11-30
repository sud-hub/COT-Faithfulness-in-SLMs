# Architecture, Algorithms, and Strategies

This document provides a deep dive into the technical architecture, technology stack, and the advanced reasoning algorithm.

## 1. System Architecture

The framework is designed with a modular architecture to facilitate easy addition of new reasoning methods and datasets. The system follows a pipeline approach:

```mermaid
graph TD
    A[Configuration (main.py)] --> B[Benchmark Engine (benchmark.py)]
    B --> C{Method Selector}
    
    C -->|Standard| D[Reasoning Methods]
    C -->|Advanced| E[Advanced Reasoning]
    C -->|SOTA| F[SOTA Reasoning]
    C -->|Hybrid| G[Hybrid Reasoning]
    
    D --> H[Ollama Interface (ollama_utils.py)]
    E --> H
    F --> H
    G --> H
    
    H --> I[Local LLM (Gemma 3)]
    I --> H
    
    H --> J[Answer Extraction & Normalization]
    J --> K[Result Logging & Checkpointing]
```

### Key Components

*   **`main.py`**: The orchestrator. It handles user configuration, method selection, and high-level execution flow.
*   **`benchmark.py`**: The core engine. It manages the evaluation loop, handles checkpointing (state persistence), and computes metrics.
*   **`ollama_utils.py`**: The abstraction layer for the LLM. It handles API communication, prompt formatting, and robust answer extraction using Regex.
*   **`data_utils.py`**: Data ingestion layer. It standardizes disparate datasets (GSM8K, StrategyQA, CommonsenseQA) into a unified format.

## 2. Technology Stack

The project is built on a lightweight, efficient stack designed for local execution.

*   **Core Language**: Python 3.10+
*   **LLM Runtime**: [Ollama](https://ollama.com/) (Local inference server)
*   **Default Model**: `gemma3:1b` (Google's lightweight open model)
*   **Key Libraries**:
    *   `requests`: For HTTP communication with Ollama.
    *   `datasets`: HuggingFace library for loading standard benchmarks.
    *   `tqdm`: For progress tracking.
    *   `re` (Regex): Heavily used for parsing unstructured LLM outputs.

## 3. Reasoning Algorithms

The framework implements a wide array of reasoning strategies, ranging from basic prompting to complex, multi-step agentic workflows.

### A. Standard Methods (`reasoning_methods.py`)
1.  **Baseline**: Direct zero-shot prompting.
2.  **Chain of Thought (CoT)**: Appends "Let's think step by step" to induce reasoning traces.
3.  **Self-Consistency (SC)**: Samples multiple outputs (temperature > 0) and takes a majority vote.
4.  **Program of Thought (PoT)**: Generates Python code to solve math problems, offloading calculation to the Python interpreter.

### B. State-of-the-Art Methods (`sota_reasoning.py`)
1.  **Least-to-Most (L2M)**: Decomposes complex problems into sub-questions (e.g., "Step 1: Find cost of apples", "Step 2: Find cost of oranges", "Step 3: Sum them").
2.  **Self-Refine**: Generates an initial answer, critiques it, and generates a refined version.
3.  **Plan-and-Solve**: Explicitly generates a plan before execution.
4.  **Tree of Thoughts (ToT-Light)**: Explores multiple reasoning "branches" (e.g., "Approach 1", "Approach 2") and selects the best path.
5.  **Analogical Prompting**: Asks the model to recall a similar solved problem to guide the current solution.

### C. Advanced & Hybrid Strategies (`hybrid_reasoning.py`)
These are custom implementations designed to maximize the performance of small models like Gemma 3 1B.

#### 1. L2M-PoT (Least-to-Most + Program-of-Thought)
*   **Concept**: Combines the decomposition of L2M with the precision of PoT.
*   **Workflow**:
    1.  Decompose problem into steps.
    2.  Generate Python code for *each* step, passing variables from previous steps.
    3.  Execute the chained code to get the final result.

#### 2. Ultimate GSM8K (Adaptive Hybrid)
*   **Concept**: An optimized pipeline for math problems that balances speed and accuracy.
*   **Workflow**:
    1.  **Adaptive Sampling**: Tries **L2M-PoT-SC** (Self-Consistency) first.
    2.  **Early Exit**: If the first 2 samples agree, it returns immediately (saving 33% compute).
    3.  **Fallback**: If no consensus, it falls back to a single robust L2M-PoT run.

#### 3. Commonsense Ensembles (`advanced_reasoning.py`)
*   **Concept**: For multiple-choice questions, single-path reasoning is often insufficient.
*   **Strategy**: Combines:
    *   **Option Analysis**: Scores each option (A-E) individually.
    *   **Contrastive Prompting**: Asks "Why is X *not* the answer?".
    *   **CoT Voting**: Standard reasoning voting.
    *   **Result**: A weighted vote determines the final answer.

## 4. Prompt Engineering Strategies

The project uses dynamic prompt formatting (`ollama_utils.py`) adapted to the question type:

*   **Math**: Few-shot examples with "####" delimiters for answers.
*   **Boolean**: Few-shot examples mapping "yes/no" to specific contexts.
*   **Multiple Choice**: Reasoning-focused prompts asking for letter extraction.

## 5. Data Flow & Evaluation

1.  **Ingestion**: `data_utils.py` loads data and normalizes it (e.g., stripping commas from numbers).
2.  **Processing**: `benchmark.py` sends prompts to the selected Algorithm.
3.  **Extraction**: `ollama_utils.py` uses regex patterns (e.g., `\boxed{}`, `Final Answer:`) to find the answer.
4.  **Normalization**: Answers are normalized (e.g., "1,234.00" -> "1234") for comparison.
5.  **Checkpointing**: Results are saved to JSON every 5 samples to prevent data loss.
