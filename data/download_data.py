"""
Data Download Script for Project 1: Dual Process Theory
Downloads REAL datasets from HuggingFace for dual process theory experiments
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Any

# Set proxy for HuggingFace downloads
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"
os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

def download_datasets():
    """Download all required datasets for dual process theory experiments"""
    
    from datasets import load_dataset
    from tqdm import tqdm
    
    # Set cache directory
    cache_dir = Path(__file__).parent / "cache"
    cache_dir.mkdir(exist_ok=True)
    os.environ["HF_HOME"] = str(cache_dir)
    
    raw_dir = Path(__file__).parent / "raw"
    processed_dir = Path(__file__).parent / "processed"
    raw_dir.mkdir(exist_ok=True)
    processed_dir.mkdir(exist_ok=True)
    
    datasets_info = {}
    
    # 1. TruthfulQA - For intuition traps (CRT-like tasks)
    print("\n" + "="*60)
    print("Downloading TruthfulQA...")
    print("="*60)
    try:
        truthfulqa = load_dataset("truthful_qa", "multiple_choice", cache_dir=str(cache_dir))
        truthfulqa_data = []
        for item in tqdm(truthfulqa["validation"], desc="Processing TruthfulQA"):
            truthfulqa_data.append({
                "question": item["question"],
                "choices": item["mc1_targets"]["choices"],
                "labels": item["mc1_targets"]["labels"],
                "correct_idx": item["mc1_targets"]["labels"].index(1) if 1 in item["mc1_targets"]["labels"] else 0,
                "source": "TruthfulQA"
            })
        
        with open(raw_dir / "truthfulqa.json", "w", encoding="utf-8") as f:
            json.dump(truthfulqa_data, f, indent=2, ensure_ascii=False)
        
        datasets_info["truthfulqa"] = {"count": len(truthfulqa_data), "path": str(raw_dir / "truthfulqa.json")}
        print(f"✓ TruthfulQA: {len(truthfulqa_data)} samples saved")
    except Exception as e:
        print(f"✗ Error downloading TruthfulQA: {e}")
    
    # 2. GSM8K - For mathematical reasoning (System 2)
    print("\n" + "="*60)
    print("Downloading GSM8K...")
    print("="*60)
    try:
        gsm8k = load_dataset("openai/gsm8k", "main", cache_dir=str(cache_dir))
        gsm8k_data = []
        for item in tqdm(gsm8k["test"], desc="Processing GSM8K"):
            answer_text = item["answer"]
            # Extract final numerical answer
            final_answer = answer_text.split("####")[-1].strip() if "####" in answer_text else ""
            gsm8k_data.append({
                "question": item["question"],
                "answer": final_answer,
                "reasoning_steps": answer_text,
                "source": "GSM8K"
            })
        
        with open(raw_dir / "gsm8k.json", "w", encoding="utf-8") as f:
            json.dump(gsm8k_data, f, indent=2, ensure_ascii=False)
        
        datasets_info["gsm8k"] = {"count": len(gsm8k_data), "path": str(raw_dir / "gsm8k.json")}
        print(f"✓ GSM8K: {len(gsm8k_data)} samples saved")
    except Exception as e:
        print(f"✗ Error downloading GSM8K: {e}")
    
    # 3. LogiQA - For logical reasoning
    print("\n" + "="*60)
    print("Downloading LogiQA...")
    print("="*60)
    try:
        logiqa = load_dataset("lucasmccabe/logiqa", cache_dir=str(cache_dir), trust_remote_code=True)
        logiqa_data = []
        for split in ["train", "validation", "test"]:
            if split in logiqa:
                for item in tqdm(logiqa[split], desc=f"Processing LogiQA {split}"):
                    logiqa_data.append({
                        "context": item["context"],
                        "question": item["query"],
                        "options": item["options"],
                        "correct_option": item["correct_option"],
                        "source": "LogiQA",
                        "split": split
                    })
        
        with open(raw_dir / "logiqa.json", "w", encoding="utf-8") as f:
            json.dump(logiqa_data, f, indent=2, ensure_ascii=False)
        
        datasets_info["logiqa"] = {"count": len(logiqa_data), "path": str(raw_dir / "logiqa.json")}
        print(f"✓ LogiQA: {len(logiqa_data)} samples saved")
    except Exception as e:
        print(f"✗ Error downloading LogiQA: {e}")
    
    # 4. BIG-Bench subsets
    print("\n" + "="*60)
    print("Downloading BIG-Bench subsets...")
    print("="*60)
    
    bigbench_tasks = [
        ("bigbench", "causal_judgment"),
        ("bigbench", "logical_deduction_five_objects"),
    ]
    
    bigbench_data = []
    for dataset_name, task_name in bigbench_tasks:
        try:
            print(f"  Loading {task_name}...")
            bb_data = load_dataset(dataset_name, task_name, cache_dir=str(cache_dir), trust_remote_code=True)
            
            # Get the available split
            available_split = list(bb_data.keys())[0] if bb_data.keys() else "default"
            
            for item in tqdm(bb_data[available_split], desc=f"Processing {task_name}"):
                bigbench_data.append({
                    "inputs": item.get("inputs", ""),
                    "targets": item.get("targets", []),
                    "multiple_choice_targets": item.get("multiple_choice_targets", []),
                    "multiple_choice_scores": item.get("multiple_choice_scores", []),
                    "task": task_name,
                    "source": "BIG-Bench"
                })
            print(f"  ✓ {task_name}: {len(bb_data[available_split])} samples")
        except Exception as e:
            print(f"  ✗ Error loading {task_name}: {e}")
    
    if bigbench_data:
        with open(raw_dir / "bigbench.json", "w", encoding="utf-8") as f:
            json.dump(bigbench_data, f, indent=2, ensure_ascii=False)
        datasets_info["bigbench"] = {"count": len(bigbench_data), "path": str(raw_dir / "bigbench.json")}
        print(f"✓ BIG-Bench total: {len(bigbench_data)} samples saved")
    
    # 5. MMLU subsets for logic and statistics
    print("\n" + "="*60)
    print("Downloading MMLU subsets...")
    print("="*60)
    
    mmlu_subsets = ["formal_logic", "elementary_mathematics", "high_school_statistics"]
    mmlu_data = []
    
    for subset in mmlu_subsets:
        try:
            print(f"  Loading MMLU {subset}...")
            mmlu = load_dataset("cais/mmlu", subset, cache_dir=str(cache_dir), trust_remote_code=True)
            
            for split in ["test", "validation"]:
                if split in mmlu:
                    for item in tqdm(mmlu[split], desc=f"Processing MMLU {subset} {split}"):
                        mmlu_data.append({
                            "question": item["question"],
                            "choices": item["choices"],
                            "answer": item["answer"],
                            "subject": subset,
                            "source": "MMLU",
                            "split": split
                        })
            print(f"  ✓ MMLU {subset}: loaded")
        except Exception as e:
            print(f"  ✗ Error loading MMLU {subset}: {e}")
    
    if mmlu_data:
        with open(raw_dir / "mmlu.json", "w", encoding="utf-8") as f:
            json.dump(mmlu_data, f, indent=2, ensure_ascii=False)
        datasets_info["mmlu"] = {"count": len(mmlu_data), "path": str(raw_dir / "mmlu.json")}
        print(f"✓ MMLU total: {len(mmlu_data)} samples saved")
    
    # Save dataset info
    with open(processed_dir / "datasets_info.json", "w", encoding="utf-8") as f:
        json.dump(datasets_info, f, indent=2)
    
    print("\n" + "="*60)
    print("Dataset Download Summary")
    print("="*60)
    for name, info in datasets_info.items():
        print(f"  {name}: {info['count']} samples")
    print(f"\nTotal datasets: {len(datasets_info)}")
    print(f"Raw data saved to: {raw_dir}")
    
    return datasets_info


def prepare_dual_process_dataset():
    """Prepare combined dataset for dual process experiments"""
    
    raw_dir = Path(__file__).parent / "raw"
    processed_dir = Path(__file__).parent / "processed"
    
    dataset = {
        "system1_tasks": [],      # Fast intuition tasks
        "system2_tasks": [],      # Deep reasoning tasks
        "conflict_tasks": [],     # Intuition vs logic conflict tasks
        "metadata": {
            "description": "Dual Process Theory Experiment Dataset",
            "version": "1.0"
        }
    }
    
    # Load TruthfulQA for conflict tasks (intuition traps)
    truthfulqa_path = raw_dir / "truthfulqa.json"
    if truthfulqa_path.exists():
        with open(truthfulqa_path, "r", encoding="utf-8") as f:
            truthfulqa = json.load(f)
        
        for i, item in enumerate(truthfulqa[:500]):  # Limit to 500
            dataset["conflict_tasks"].append({
                "id": f"truthful_{i:04d}",
                "question": item["question"],
                "options": item["choices"],
                "correct_answer": item["correct_idx"],
                "task_type": "intuition_trap",
                "source": "TruthfulQA"
            })
    
    # Load GSM8K for System 2 tasks
    gsm8k_path = raw_dir / "gsm8k.json"
    if gsm8k_path.exists():
        with open(gsm8k_path, "r", encoding="utf-8") as f:
            gsm8k = json.load(f)
        
        for i, item in enumerate(gsm8k[:500]):  # Limit to 500
            dataset["system2_tasks"].append({
                "id": f"gsm8k_{i:04d}",
                "question": item["question"],
                "correct_answer": item["answer"],
                "reasoning_steps": item["reasoning_steps"],
                "task_type": "math_reasoning",
                "source": "GSM8K"
            })
    
    # Load LogiQA for System 2 tasks
    logiqa_path = raw_dir / "logiqa.json"
    if logiqa_path.exists():
        with open(logiqa_path, "r", encoding="utf-8") as f:
            logiqa = json.load(f)
        
        test_logiqa = [item for item in logiqa if item.get("split") == "test"][:500]
        for i, item in enumerate(test_logiqa):
            dataset["system2_tasks"].append({
                "id": f"logiqa_{i:04d}",
                "context": item["context"],
                "question": item["question"],
                "options": item["options"],
                "correct_answer": item["correct_option"],
                "task_type": "logical_reasoning",
                "source": "LogiQA"
            })
    
    # Load BIG-Bench for mixed tasks
    bigbench_path = raw_dir / "bigbench.json"
    if bigbench_path.exists():
        with open(bigbench_path, "r", encoding="utf-8") as f:
            bigbench = json.load(f)
        
        for i, item in enumerate(bigbench[:300]):
            task_type = "causal_judgment" if "causal" in item.get("task", "") else "logical_deduction"
            target_list = "system1_tasks" if task_type == "causal_judgment" else "system2_tasks"
            
            dataset[target_list].append({
                "id": f"bigbench_{i:04d}",
                "question": item["inputs"],
                "options": item.get("multiple_choice_targets", []),
                "correct_answer": item.get("targets", []),
                "task_type": task_type,
                "source": "BIG-Bench"
            })
    
    # Load MMLU for System 1 (simple) and System 2 (complex) tasks
    mmlu_path = raw_dir / "mmlu.json"
    if mmlu_path.exists():
        with open(mmlu_path, "r", encoding="utf-8") as f:
            mmlu = json.load(f)
        
        for i, item in enumerate(mmlu[:400]):
            # Elementary math goes to System 1, formal logic to System 2
            if item.get("subject") == "elementary_mathematics":
                target_list = "system1_tasks"
                task_type = "simple_math"
            else:
                target_list = "system2_tasks"
                task_type = "formal_logic"
            
            dataset[target_list].append({
                "id": f"mmlu_{i:04d}",
                "question": item["question"],
                "options": item["choices"],
                "correct_answer": item["answer"],
                "subject": item.get("subject", ""),
                "task_type": task_type,
                "source": "MMLU"
            })
    
    # Save processed dataset
    with open(processed_dir / "dual_process_dataset.json", "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*60)
    print("Processed Dataset Summary")
    print("="*60)
    print(f"  System 1 tasks: {len(dataset['system1_tasks'])}")
    print(f"  System 2 tasks: {len(dataset['system2_tasks'])}")
    print(f"  Conflict tasks: {len(dataset['conflict_tasks'])}")
    print(f"\nSaved to: {processed_dir / 'dual_process_dataset.json'}")
    
    return dataset


if __name__ == "__main__":
    print("="*60)
    print("Project 1: Dual Process Theory - Data Download")
    print("="*60)
    
    # Download raw datasets
    download_datasets()
    
    # Prepare processed dataset
    prepare_dual_process_dataset()
    
    print("\n✓ Data download and processing complete!")
