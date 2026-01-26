# Dual Process Theory in Large Language Models

This repository contains the code and experiments for investigating dual process theory (System 1 vs System 2 thinking) in Large Language Models.

## Overview

Dual Process Theory, proposed by Kahneman and Tversky, distinguishes between two modes of cognitive processing:

- **System 1**: Fast, intuitive, automatic thinking
- **System 2**: Slow, deliberate, analytical thinking

This project implements and evaluates these two cognitive systems using LLMs, comparing their performance on various cognitive tasks and benchmarking against human cognitive psychology data.

## Project Structure

```
├── src/
│   ├── systems/           # System 1 and System 2 implementations
│   │   ├── system1.py     # Fast intuitive thinking system
│   │   ├── system2.py     # Slow analytical thinking system
│   │   └── actr_buffers.py # ACT-R cognitive architecture components
│   ├── tasks/             # Task definitions and loaders
│   │   ├── crt_tasks.py   # Cognitive Reflection Test tasks
│   │   ├── logical_reasoning.py
│   │   ├── math_reasoning.py
│   │   └── commonsense_tasks.py
│   ├── evaluation/        # Evaluation metrics and experiment runners
│   │   ├── evaluator.py
│   │   ├── metrics.py
│   │   ├── visualizer.py
│   │   └── experiment_runner.py
│   ├── run_experiment.py  # Main experiment script
│   └── run_parallel_experiment.py
├── results/               # Experiment results and figures
```

## Key Features

- **Dual System Implementation**: System 1 uses smaller/faster models with high temperature, while System 2 uses larger models with chain-of-thought reasoning
- **ACT-R Integration**: Cognitive architecture components (Sensory Buffer, Goal Buffer, Declarative Memory, Production System)
- **Multiple Task Categories**: 
  - System 1 tasks (commonsense, pattern recognition)
  - System 2 tasks (math, logic)
  - Conflict tasks (CRT-style cognitive traps)
- **Human Benchmark Comparison**: Compares LLM performance with human cognitive psychology data
- **Statistical Analysis**: Effect sizes, confidence intervals, paired t-tests

## Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/dual-process-llm.git
cd dual-process-llm

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Configuration

Create an `api_config.py` file in the project root with your OpenAI API configuration:

```python
from openai import OpenAI

def get_openai_client():
    return OpenAI(api_key="your-api-key")

def get_model_config():
    return {
        "system1_model": "gpt-4o-mini",
        "system2_model": "gpt-4o"
    }
```

## Usage

### Quick Test
```bash
python src/run_experiment.py --experiment quick_test --n_samples 10
```

### Full Experiment
```bash
python src/run_experiment.py --experiment full --n_samples 100
```

### Paper-Level Experiments
```bash
# Quick validation
python src/run_experiment.py --experiment paper --mode quick_validation

# Full paper run (ICLR/NeurIPS standard)
python src/run_experiment.py --experiment paper --mode full
```

### Ablation Study
```bash
python src/run_experiment.py --experiment ablation --n_samples 50
```

### Parallel Execution
```bash
python src/run_parallel_experiment.py --mode paper --workers 4
```

## Experiment Types

| Experiment | Description | Samples |
|------------|-------------|---------|
| `quick_test` | Fast validation | 10/category |
| `full` | Complete experiment | 100/category |
| `ablation` | Variable isolation | 50/category |
| `category` | Per-category analysis | 100/category |
| `stepped` | Factor contribution | 30/category |
| `paper` | Publication-ready | 200/category |

## Results

The experiments generate:
- JSON results with full metrics
- Publication-quality figures (PNG)
- LaTeX tables for papers
- Human benchmark comparisons

## Human Benchmarks

We compare against established cognitive psychology benchmarks:
- **CRT** (Frederick, 2005): Cognitive Reflection Test
- **Base Rate Neglect** (Kahneman & Tversky, 1973)
- **Logical Reasoning** (Evans et al., 1983)
- **Framing Effects** (Tversky & Kahneman, 1981)

## Citation

If you use this code in your research, please cite:

```bibtex
@article{anonymous2026dual,
  title={Dual Process Theory in Large Language Models: 
         Investigating System 1 and System 2 Thinking},
  author={Anonymous},
  journal={Under Review},
  year={2026}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Kahneman, D. (2011). Thinking, Fast and Slow
- Frederick, S. (2005). Cognitive Reflection and Decision Making
- Anderson, J. R. (2007). How Can the Human Mind Occur in the Physical Universe?
