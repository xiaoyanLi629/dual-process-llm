#!/bin/bash
# Project 1: Dual Process Theory - Run All Experiments
# This script runs all experiments sequentially

cd /root/autodl-fs/LLM-Cognition/idea1_dual_process/src

echo "============================================================"
echo "Project 1: Dual Process Theory - All Experiments"
echo "============================================================"
echo "Start Time: $(date)"
echo ""

# 1. Quick Test
echo "=== [1/5] Quick Test (n=10) ==="
python run_experiment.py --experiment quick_test --n_samples 10
echo "Quick test completed at $(date)"
echo ""

# 2. Full Experiment
echo "=== [2/5] Full Experiment (n=50) ==="
python run_experiment.py --experiment full --n_samples 50
echo "Full experiment completed at $(date)"
echo ""

# 3. Ablation Study
echo "=== [3/5] Ablation Study (n=30) ==="
python run_experiment.py --experiment ablation --n_samples 30
echo "Ablation study completed at $(date)"
echo ""

# 4. Category Analysis
echo "=== [4/5] Category Analysis (n=30) ==="
python run_experiment.py --experiment category --n_samples 30
echo "Category analysis completed at $(date)"
echo ""

# 5. Stepped Experiment
echo "=== [5/5] Stepped Experiment (n=20) ==="
python run_experiment.py --experiment stepped --n_samples 20
echo "Stepped experiment completed at $(date)"
echo ""

echo "============================================================"
echo "All experiments completed!"
echo "End Time: $(date)"
echo "============================================================"

# List results
echo ""
echo "Results saved in:"
ls -la ../results/
