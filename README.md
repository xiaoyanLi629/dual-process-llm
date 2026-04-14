# Dual Process Theory in Large Language Models

This repository contains the code and experiments for investigating dual process theory (System 1 vs System 2 thinking) in Large Language Models. Target venue: **IEEE BIBM 2026**.

## Overview

Dual Process Theory, proposed by Kahneman and Tversky, distinguishes between two modes of cognitive processing:

- **System 1**: Fast, intuitive, automatic thinking
- **System 2**: Slow, deliberate, analytical thinking

This project implements and evaluates these two cognitive systems using LLMs, comparing their performance on various cognitive tasks and benchmarking against human cognitive psychology data. Key contributions include a **2x2x2 factorial ablation** isolating model size, temperature, and prompting strategy, **multi-model validation** across 4 LLM families, and **novel conflict tasks** resistant to training data memorization.

## Project Structure

```
├── src/
│   ├── systems/           # System 1 and System 2 implementations
│   │   ├── system1.py     # Fast intuitive thinking system
│   │   ├── system2.py     # Slow analytical thinking system
│   │   └── actr_buffers.py # ACT-R cognitive architecture components
│   ├── tasks/             # Task definitions and loaders
│   │   ├── crt_tasks.py   # Cognitive Reflection Test tasks
│   │   ├── novel_conflict_tasks.py  # Novel conflict tasks (50+)
│   │   ├── logical_reasoning.py
│   │   ├── math_reasoning.py
│   │   └── commonsense_tasks.py
│   ├── experiments/       # BIBM 2026 experiment scripts
│   │   ├── ablation_factorial.py    # 2x2x2 factorial ablation
│   │   ├── multi_model.py           # Cross-model validation
│   │   ├── human_comparison.py      # Human-LLM pattern comparison
│   │   ├── statistical_analysis.py  # Comprehensive statistics
│   │   └── run_all_bibm.py          # Master experiment runner
│   ├── evaluation/        # Evaluation metrics and experiment runners
│   │   ├── evaluator.py
│   │   ├── metrics.py
│   │   ├── visualizer.py
│   │   └── experiment_runner.py
│   ├── visualization/     # Publication figure generation
│   │   └── bibm_figures.py
│   ├── run_experiment.py  # Original experiment script
│   └── run_parallel_experiment.py
├── IEEE_manuscript/       # IEEE BIBM 2026 paper (target venue)
├── data/                  # Task datasets
├── results/               # Experiment results and figures
└── Do_Large_Language_Models_Think_Fast_and_Slow/  # CogSci submission (archived)
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

### BIBM 2026 Experiments (Recommended)

```bash
# Dry run — verify setup (1 sample per condition, no real cost)
python -m src.experiments.run_all_bibm --all --dry_run

# Run factorial ablation only
python -m src.experiments.run_all_bibm --experiment ablation --n_samples 100

# Run multi-model validation
python -m src.experiments.run_all_bibm --experiment multi_model --n_samples 100

# Run all experiments
python -m src.experiments.run_all_bibm --all --n_samples 100

# Post-hoc analysis only (no API calls)
python -m src.experiments.run_all_bibm --analyze_only --input results/bibm_2026/
```

### Generate Publication Figures

```bash
# From experiment results
python -m src.visualization.bibm_figures --input results/bibm_2026/ --output IEEE_manuscript/figures/

# With sample data (for layout testing)
python -m src.visualization.bibm_figures --sample --output IEEE_manuscript/figures/
```

### Legacy Experiments

```bash
python src/run_experiment.py --experiment quick_test --n_samples 10
python src/run_experiment.py --experiment full --n_samples 100
python src/run_parallel_experiment.py --mode paper --workers 4
```

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
@inproceedings{anonymous2026dual,
  title={Do Large Language Models Think Fast and Slow? 
         Simulating Dual-Process Cognition for Biomedical Decision Support},
  author={Anonymous},
  booktitle={IEEE International Conference on Bioinformatics and Biomedicine (BIBM)},
  year={2026}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Kahneman, D. (2011). Thinking, Fast and Slow
- Frederick, S. (2005). Cognitive Reflection and Decision Making
- Anderson, J. R. (2007). How Can the Human Mind Occur in the Physical Universe?
