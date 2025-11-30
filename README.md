This is an advanced benchmarking and reasoning framework designed to evaluate and enhance the capabilities of small Large Language Models (LLMs), such as **Gemma 3 1B**, on complex reasoning tasks. 

It implements a wide range of State-of-the-Art (SOTA) algorithms—from Chain-of-Thought to Hybrid Program-of-Thought—to push the limits of what small models can achieve.

## Key Features

*   **Modular Architecture**: Easily plug-and-play different reasoning methods and datasets.
*   **Advanced Algorithms**: Includes implementations of Least-to-Most, Tree of Thoughts, Self-Refine, and custom Hybrid strategies.
*   **Robust Evaluation**: Features automatic checkpointing, time estimation, and detailed CSV reporting.
*   **Local-First**: Built for **Ollama**, ensuring privacy and zero-cost execution.
*   **Specialized Hybrids**: Contains unique strategies like `Ultimate-GSM8K` (Adaptive L2M-PoT) optimized for small model performance.

## Tech Stack

*   **Language**: Python 3.10+
*   **LLM Engine**: [Ollama](https://ollama.com/)
*   **Model**: `gemma3:1b` (Configurable)
*   **Datasets**: GSM8K (Math), StrategyQA (Reasoning), CommonsenseQA (Knowledge)

## Repository Structure

```text
STC/
├── main.py                     # Entry point for the benchmark
├── benchmark.py                # Core evaluation engine
├── compare_prompt_strategies.py # Diagnostic tool for prompt testing
├── requirements.txt            # Python dependencies
├── docs/
│   └── ARCHITECTURE_AND_STRATEGIES.md  # Detailed technical docs
├── data/                       # Dataset storage
├── results/                    # CSV outputs and checkpoints
└── src/
    ├── reasoning_methods.py    # Standard methods (CoT, SC, PoT)
    ├── sota_reasoning.py       # SOTA methods (L2M, ToT, Self-Refine)
    ├── advanced_reasoning.py   # MC/Boolean specific strategies
    ├── hybrid_reasoning.py     # Custom hybrid pipelines
    ├── ollama_utils.py         # LLM interface & answer extraction
    └── data_utils.py           # Data loading & normalization
```

## Quick Start

### 1. Prerequisites
*   Install [Ollama](https://ollama.com/).
*   Pull the default model:
    ```bash
    ollama pull gemma3:1b
    ```

### 2. Installation
Clone the repository and install dependencies:
```bash
pip install -r requirements.txt
```

### 3. Running the Benchmark
To start a full evaluation or a sample run:
```bash
python main.py
```
*Follow the on-screen prompts to select evaluation modes.*

### 4. Comparing Strategies
To quickly test which prompting strategy works best for a specific model:
```bash
python compare_prompt_strategies.py
```

## Supported Algorithms

For a deep dive into how these work, see [docs/ARCHITECTURE_AND_STRATEGIES.md](docs/ARCHITECTURE_AND_STRATEGIES.md).

| Category | Algorithms |
|----------|------------|
| **Standard** | Baseline, Chain of Thought (CoT), Self-Consistency (SC) |
| **Code-Based** | Program of Thought (PoT), PoT-SC |
| **Decomposition** | Least-to-Most (L2M), Plan-and-Solve |
| **Refinement** | Self-Refine, Progressive Hint |
| **Exploration** | Tree of Thoughts (Light), Multi-Persona |
| **Hybrid (Custom)** | L2M-PoT, Ultimate-GSM8K, Ensemble-Commonsense |

## Results

Results are saved automatically to the `results/` directory in CSV format.
*   `results/{dataset}_{method}.csv`: Full detailed logs.
*   `results/evaluation_summary.txt`: High-level summary of the last run.
