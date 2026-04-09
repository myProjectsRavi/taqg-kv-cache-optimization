#!/usr/bin/env python3
"""
TAQG Paper — Complete Experimental Validation Suite (v2 — Fixed)
================================================================
Designed for Google Colab FREE tier (T4 16GB GPU).

FIXES over v1:
  - Uses chat template so model emits <think>...</think> markers
  - Handles DynamicCache (transformers >= 4.40) correctly
  - Single-pass: generates each prompt ONCE, runs all experiments
  - NaN guards in all computations
  - Runtime: ~50 minutes (down from ~4 hours)

Model: DeepSeek-R1-Distill-Qwen-1.5B (fits on T4 with float16)
Dataset: GSM8K test set (50 samples)

Usage (Google Colab):
  1. Open new Colab notebook
  2. Set runtime to T4 GPU
  3. Paste this entire script into a cell
  4. Run — takes ~50 minutes total

Author: Raviteja Nekkalapu
License: MIT
"""

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # Better CUDA error reporting
import traceback
import json
import time
import re
import warnings
import gc
from pathlib import Path
from collections import defaultdict

import numpy as np

# ============================================================
# SECTION 0: Setup & Installation
# ============================================================

def install_dependencies():
    os.system("pip install -q transformers>=4.40.0 accelerate>=0.28.0 datasets torch matplotlib seaborn scipy tqdm")
    print("✅ Dependencies installed.")

install_dependencies()

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# ============================================================
# SECTION 1: Configuration
# ============================================================

CONFIG = {
    "model_name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "dtype": torch.float32,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "dataset": "openai/gsm8k",
    "dataset_split": "test",
    "num_samples": 50,
    "max_new_tokens": 1024,
    "think_end_markers": ["</think>", "\n\nFinal Answer:", "\n\n**Final Answer", "\\boxed{"],
    "entropy_window_size": 10,
    "tau_sweep": np.arange(0.05, 1.55, 0.05),
    "output_dir": "taqg_results",
    "figures_dir": "taqg_results/figures",
    "seed": 42,
}

os.makedirs(CONFIG["output_dir"], exist_ok=True)
os.makedirs(CONFIG["figures_dir"], exist_ok=True)
torch.manual_seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])

print(f"🖥️  Device: {CONFIG['device']}")
if CONFIG["device"] == "cuda":
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# ============================================================
# SECTION 2: Model & Data Loading
# ============================================================

def load_model_and_tokenizer():
    print(f"\n📦 Loading {CONFIG['model_name']}...")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        CONFIG["model_name"],
        torch_dtype=CONFIG["dtype"],
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()

    # Verify tokenizer/model vocab alignment
    print(f"   Tokenizer vocab size: {tokenizer.vocab_size}")
    print(f"   Model vocab size:     {model.config.vocab_size}")
    if hasattr(model, 'lm_head'):
        print(f"   LM head output dim:   {model.lm_head.out_features}")

    print(f"✅ Model loaded. Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M")
    return model, tokenizer


def load_gsm8k_prompts():
    print(f"\n📊 Loading GSM8K ({CONFIG['num_samples']} samples)...")
    ds = load_dataset(CONFIG["dataset"], "main", split=CONFIG["dataset_split"])
    ds = ds.select(range(min(CONFIG["num_samples"], len(ds))))

    prompts = []
    ground_truths = []
    for item in ds:
        prompt = f"Solve this math problem step by step, showing your reasoning:\n\n{item['question']}\n\nThink carefully before giving your final answer."
        prompts.append(prompt)
        answer = item["answer"].split("####")[-1].strip()
        ground_truths.append(answer)

    print(f"✅ Loaded {len(prompts)} prompts.")
    return prompts, ground_truths


# ============================================================
# SECTION 3: Generation with KV Cache Extraction
# ============================================================

def find_think_boundary(generated_text, token_ids, tokenizer):
    for marker in CONFIG["think_end_markers"]:
        idx = generated_text.find(marker)
        if idx != -1:
            prefix = generated_text[:idx + len(marker)]
            prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
            return len(prefix_tokens), "marker"
    return int(len(token_ids) * 0.80), "heuristic"


def extract_kv_cache(past_key_values, input_len):
    """Extract KV cache handling DynamicCache across all transformers versions."""
    kv_cache = {}
    if past_key_values is None:
        return kv_cache

    # Approach 1: DynamicCache.key_cache (public attribute)
    if hasattr(past_key_values, 'key_cache') and hasattr(past_key_values, 'value_cache'):
        keys_list = past_key_values.key_cache
        vals_list = past_key_values.value_cache
        for layer_idx in range(len(keys_list)):
            k, v = keys_list[layer_idx], vals_list[layer_idx]
            kv_cache[layer_idx] = (k[:, :, input_len:, :].cpu(), v[:, :, input_len:, :].cpu())
        return kv_cache

    # Approach 2: DynamicCache._key_cache (private attribute)
    if hasattr(past_key_values, '_key_cache') and hasattr(past_key_values, '_value_cache'):
        keys_list = past_key_values._key_cache
        vals_list = past_key_values._value_cache
        for layer_idx in range(len(keys_list)):
            k, v = keys_list[layer_idx], vals_list[layer_idx]
            kv_cache[layer_idx] = (k[:, :, input_len:, :].cpu(), v[:, :, input_len:, :].cpu())
        return kv_cache

    # Approach 3: to_legacy_cache() method
    if hasattr(past_key_values, 'to_legacy_cache'):
        legacy = past_key_values.to_legacy_cache()
        for layer_idx, (k, v) in enumerate(legacy):
            kv_cache[layer_idx] = (k[:, :, input_len:, :].cpu(), v[:, :, input_len:, :].cpu())
        return kv_cache

    # Approach 4: Iterate (DynamicCache.__iter__ yields (key, value) tuples)
    try:
        for layer_idx, layer_data in enumerate(past_key_values):
            if isinstance(layer_data, (tuple, list)) and len(layer_data) >= 2:
                k, v = layer_data[0], layer_data[1]
            elif isinstance(layer_data, torch.Tensor):
                continue  # Skip non-tuple layers
            else:
                continue
            kv_cache[layer_idx] = (k[:, :, input_len:, :].cpu(), v[:, :, input_len:, :].cpu())
        if kv_cache:
            return kv_cache
    except Exception:
        pass

    # Debug: print what attributes the object actually has
    print(f"    ⚠️ KV cache type: {type(past_key_values).__name__}")
    cache_attrs = [a for a in dir(past_key_values) if 'cache' in a.lower() or 'key' in a.lower()]
    print(f"    ⚠️ Available cache attrs: {cache_attrs}")
    return kv_cache


def generate_with_kv_extraction(model, tokenizer, prompt, max_new_tokens=None):
    if max_new_tokens is None:
        max_new_tokens = CONFIG["max_new_tokens"]

    # Use chat template so model emits <think>...</think> markers
    # tokenize=True returns List[int] with correct special token IDs
    # (tokenize=False + re-tokenizing breaks special tokens like <|im_start|>)
    try:
        messages = [{"role": "user", "content": prompt}]
        token_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        )
        input_ids = torch.tensor([token_ids], dtype=torch.long).to(CONFIG["device"])
    except Exception:
        inputs = tokenizer(prompt, return_tensors="pt").to(CONFIG["device"])
        input_ids = inputs["input_ids"]

    input_len = input_ids.shape[1]
    attention_mask = torch.ones_like(input_ids)

    # Sampling with float32 model (float16 caused NaN → CUDA crash;
    # greedy decoding produces degenerate output for this model)
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.6,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,
        )

    generated_ids = outputs.sequences[0, input_len:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # Per-token entropy — cast to float32 for numerical stability
    token_entropies = []
    for score in outputs.scores:
        probs = F.softmax(score[0].float(), dim=-1).clamp(min=1e-10)
        entropy = -(probs * probs.log()).sum().item()
        if np.isnan(entropy) or np.isinf(entropy):
            entropy = 0.0
        token_entropies.append(entropy)

    # Forward pass to get KV cache
    full_ids = outputs.sequences[:, :input_len + len(generated_ids)]
    with torch.no_grad():
        kv_outputs = model(input_ids=full_ids, use_cache=True)

    # === FIX #2: Handle DynamicCache properly ===
    kv_cache = extract_kv_cache(kv_outputs.past_key_values, input_len)

    t_star, method = find_think_boundary(generated_text, generated_ids.tolist(), tokenizer)

    return {
        "generated_text": generated_text,
        "token_ids": generated_ids.tolist(),
        "kv_cache": kv_cache,
        "token_entropies": token_entropies,
        "t_star": t_star,
        "detection_method": method,
        "num_tokens": len(generated_ids),
        "input_len": input_len,
    }


# ============================================================
# EXPERIMENT 1: Measure Pairwise Cosine Redundancy ρ
# ============================================================

def compute_pairwise_cosine_redundancy(kv_cache, t_star, layers_to_sample=None):
    per_layer_results = {}
    all_rho_think = []
    all_rho_answer = []

    if layers_to_sample is None:
        layers_to_sample = list(kv_cache.keys())

    for layer_idx in layers_to_sample:
        keys, _ = kv_cache[layer_idx]
        k = keys[0].mean(dim=0).float()  # (seq_len, head_dim)

        if k.shape[0] < 4 or t_star < 2:
            continue

        # === FIX #3: NaN guard — skip zero vectors ===
        norms = k.norm(dim=-1)
        valid = norms > 1e-8
        if valid.sum() < 4:
            continue

        k_normed = F.normalize(k, dim=-1)

        k_think = k_normed[:t_star]
        k_answer = k_normed[t_star:]

        # Filter out any NaN rows
        k_think = k_think[~torch.isnan(k_think).any(dim=-1)]
        k_answer = k_answer[~torch.isnan(k_answer).any(dim=-1)]

        rho_think = 0.0
        rho_think_std = 0.0
        if k_think.shape[0] >= 2:
            sim_think = torch.mm(k_think, k_think.t())
            mask = torch.triu(torch.ones_like(sim_think, dtype=torch.bool), diagonal=1)
            vals = sim_think[mask]
            if vals.numel() > 0 and not torch.isnan(vals).any():
                rho_think = vals.mean().item()
                rho_think_std = vals.std().item()

        rho_answer = 0.0
        rho_answer_std = 0.0
        if k_answer.shape[0] >= 2:
            sim_answer = torch.mm(k_answer, k_answer.t())
            mask = torch.triu(torch.ones_like(sim_answer, dtype=torch.bool), diagonal=1)
            vals = sim_answer[mask]
            if vals.numel() > 0 and not torch.isnan(vals).any():
                rho_answer = vals.mean().item()
                rho_answer_std = vals.std().item()

        per_layer_results[layer_idx] = {
            "rho_think": rho_think, "rho_think_std": rho_think_std,
            "rho_answer": rho_answer, "rho_answer_std": rho_answer_std,
            "n_think": int(k_think.shape[0]), "n_answer": int(k_answer.shape[0]),
        }
        all_rho_think.append(rho_think)
        all_rho_answer.append(rho_answer)

    aggregate = {
        "mean_rho_think": float(np.mean(all_rho_think)) if all_rho_think else 0.0,
        "std_rho_think": float(np.std(all_rho_think)) if all_rho_think else 0.0,
        "mean_rho_answer": float(np.mean(all_rho_answer)) if all_rho_answer else 0.0,
        "std_rho_answer": float(np.std(all_rho_answer)) if all_rho_answer else 0.0,
    }
    return per_layer_results, aggregate


# ============================================================
# EXPERIMENT 2: Entropy-Based Phase Detection (τ sweep)
# ============================================================

def entropy_phase_detection(entropies, window_size, tau):
    T = len(entropies)
    if T <= window_size:
        return T
    for t in range(window_size, T - window_size):
        h_prev = np.mean(entropies[max(0, t - window_size):t])
        h_curr = np.mean(entropies[t:min(t + window_size, T)])
        delta_h = h_prev - h_curr
        if delta_h > tau:
            return t
    return T


# ============================================================
# EXPERIMENT 3: Simulated Quantization
# ============================================================

def simulate_quantization(tensor, bits):
    if bits >= 16:
        return tensor
    max_val = tensor.abs().max()
    if max_val == 0:
        return tensor
    n_levels = 2 ** bits - 1
    scale = max_val / (n_levels // 2)
    quantized = torch.round(tensor / scale).clamp(-n_levels // 2, n_levels // 2) * scale
    return quantized


def compute_attention_distortion(keys_orig, keys_quant, query):
    scores_orig = torch.matmul(query, keys_orig.t()) / np.sqrt(keys_orig.shape[-1])
    scores_quant = torch.matmul(query, keys_quant.t()) / np.sqrt(keys_quant.shape[-1])
    p = F.softmax(scores_orig, dim=-1).clamp(min=1e-10)
    q = F.softmax(scores_quant, dim=-1).clamp(min=1e-10)
    kl = (p * (p.log() - q.log())).sum().item()
    if np.isnan(kl) or np.isinf(kl):
        return 0.0
    return kl


# ============================================================
# EXPERIMENT 4: Answer extraction
# ============================================================

def extract_numeric_answer(text):
    boxed_match = re.search(r'\\boxed\{([^}]+)\}', text)
    if boxed_match:
        return boxed_match.group(1).strip().replace(",", "")
    final_match = re.search(r'(?:final answer|the answer|answer)\s*(?:is|:)\s*\$?([0-9,.\-]+)', text, re.IGNORECASE)
    if final_match:
        return final_match.group(1).strip().replace(",", "")
    numbers = re.findall(r'[-]?\d+(?:,\d{3})*(?:\.\d+)?', text)
    if numbers:
        return numbers[-1].replace(",", "")
    return None


# ============================================================
# SINGLE-PASS PROCESSING (generates each prompt ONCE)
# ============================================================

BIT_CONFIGS = [
    {"name": "Uniform 4-bit", "b_think": 4, "b_answer": 4},
    {"name": "Uniform 3-bit", "b_think": 3, "b_answer": 3},
    {"name": "Uniform 2-bit", "b_think": 2, "b_answer": 2},
    # Flipped allocations: lower bits for the more redundant Answer phase
    {"name": "TA-Flipped (4/3)", "b_think": 4, "b_answer": 3},
    {"name": "TA-Flipped (4/2)", "b_think": 4, "b_answer": 2},
    {"name": "TA-TurboQuant (3/4)", "b_think": 3, "b_answer": 4},
]



def process_sample(result, ground_truth):
    """Run all experiment analyses on a single generated result."""
    data = {
        "t_star": result["t_star"],
        "num_tokens": result["num_tokens"],
        "detection_method": result["detection_method"],
        "alpha": result["t_star"] / max(result["num_tokens"], 1),
        "generated_text": result["generated_text"],
    }

    # --- Exp 1: ρ measurement ---
    if result["kv_cache"]:
        num_layers = len(result["kv_cache"])
        layer_indices = list(range(0, num_layers, max(1, num_layers // 8)))
        per_layer, agg = compute_pairwise_cosine_redundancy(
            result["kv_cache"], result["t_star"], layer_indices
        )
        data["exp1_per_layer"] = {str(k): v for k, v in per_layer.items()}
        data["exp1_agg"] = agg
    else:
        data["exp1_per_layer"] = {}
        data["exp1_agg"] = {"mean_rho_think": 0, "std_rho_think": 0,
                            "mean_rho_answer": 0, "std_rho_answer": 0}

    # --- Exp 2: store entropies for later τ sweep ---
    data["entropies"] = result["token_entropies"]

    # --- Exp 3: quantization distortion ---
    data["exp3"] = {}
    if result["kv_cache"]:
        t_star = result["t_star"]
        n_total = result["num_tokens"]
        mid_layer = len(result["kv_cache"]) // 2
        keys, _ = result["kv_cache"][mid_layer]
        k = keys[0].mean(dim=0).float()

        if k.shape[0] >= 4:
            k_answer_raw = k[t_star:]
            if k_answer_raw.shape[0] > 0:
                query = F.normalize(k_answer_raw[0:1], dim=-1)
            else:
                query = F.normalize(torch.randn(1, k.shape[-1]), dim=-1)

            for cfg in BIT_CONFIGS:
                k_think = k[:t_star]
                k_answer = k[t_star:]
                k_think_q = simulate_quantization(k_think, cfg["b_think"])
                k_answer_q = simulate_quantization(k_answer, cfg["b_answer"])
                k_quantized = torch.cat([k_think_q, k_answer_q], dim=0)
                kl = compute_attention_distortion(k, k_quantized, query)

                eff_bits = (t_star * cfg["b_think"] + (n_total - t_star) * cfg["b_answer"]) / max(n_total, 1)
                compression = 16.0 / max(eff_bits, 0.1)

                data["exp3"][cfg["name"]] = {"kl": kl, "compression": compression}

    # --- Exp 4: accuracy ---
    pred = extract_numeric_answer(result["generated_text"])
    data["exp4_pred"] = pred
    data["exp4_correct"] = False
    if pred is not None and ground_truth is not None:
        try:
            if abs(float(pred) - float(ground_truth)) < 1e-3:
                data["exp4_correct"] = True
        except ValueError:
            if pred.strip() == ground_truth.strip():
                data["exp4_correct"] = True

    return data


# ============================================================
# AGGREGATION & REPORTING
# ============================================================

def aggregate_and_report(all_data):
    """Aggregate per-sample data into final experiment results."""

    # ========== EXPERIMENT 1 ==========
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: Pairwise Cosine Redundancy ρ")
    print("=" * 60)

    rho_think_vals = [d["exp1_agg"]["mean_rho_think"] for d in all_data if d["exp1_agg"]["mean_rho_think"] != 0]
    rho_answer_vals = [d["exp1_agg"]["mean_rho_answer"] for d in all_data if d["exp1_agg"]["mean_rho_answer"] != 0]
    alpha_vals = [d["alpha"] for d in all_data]

    exp1 = {
        "rho_think_mean": float(np.mean(rho_think_vals)) if rho_think_vals else 0.0,
        "rho_think_std": float(np.std(rho_think_vals)) if rho_think_vals else 0.0,
        "rho_think_median": float(np.median(rho_think_vals)) if rho_think_vals else 0.0,
        "rho_think_q25": float(np.percentile(rho_think_vals, 25)) if rho_think_vals else 0.0,
        "rho_think_q75": float(np.percentile(rho_think_vals, 75)) if rho_think_vals else 0.0,
        "rho_answer_mean": float(np.mean(rho_answer_vals)) if rho_answer_vals else 0.0,
        "rho_answer_std": float(np.std(rho_answer_vals)) if rho_answer_vals else 0.0,
        "rho_answer_median": float(np.median(rho_answer_vals)) if rho_answer_vals else 0.0,
        "alpha_mean": float(np.mean(alpha_vals)),
        "alpha_std": float(np.std(alpha_vals)),
        "num_valid_samples": len(rho_think_vals),
        "total_samples": len(all_data),
    }

    print(f"\n{'─' * 50}")
    print(f"  ρ_think  = {exp1['rho_think_mean']:.4f} ± {exp1['rho_think_std']:.4f}")
    print(f"           median = {exp1['rho_think_median']:.4f}, IQR = [{exp1['rho_think_q25']:.4f}, {exp1['rho_think_q75']:.4f}]")
    print(f"  ρ_answer = {exp1['rho_answer_mean']:.4f} ± {exp1['rho_answer_std']:.4f}")
    print(f"  α (think fraction) = {exp1['alpha_mean']:.4f} ± {exp1['alpha_std']:.4f}")
    print(f"  Valid samples: {exp1['num_valid_samples']}/{exp1['total_samples']}")

    marker_count = sum(1 for d in all_data if d["detection_method"] == "marker")
    print(f"  Marker-detected boundaries: {marker_count}/{len(all_data)}")
    print(f"{'─' * 50}")

    if exp1['rho_think_mean'] >= 0.5:
        print(f"  ✅ ρ_think ≥ 0.5 — SUPPORTS paper's assumption")
    else:
        print(f"  ⚠️  ρ_think < 0.5 — paper claims may need revision")
    if exp1['rho_think_mean'] > exp1['rho_answer_mean'] + 0.05:
        print(f"  ✅ ρ_think >> ρ_answer — Confirms think-answer asymmetry")

    save_json(exp1, "experiment1_rho.json")

    # ========== EXPERIMENT 2 ==========
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Entropy-Based Phase Detection")
    print("=" * 60)

    # Use samples with marker-based boundaries as ground truth
    marker_samples = [d for d in all_data if d["detection_method"] == "marker" and len(d["entropies"]) > 20]

    if not marker_samples:
        print("  ⚠️  No marker-based boundaries. Using heuristic boundaries for τ sweep.")
        marker_samples = [d for d in all_data if len(d["entropies"]) > 20]

    tau_values = CONFIG["tau_sweep"]
    window = CONFIG["entropy_window_size"]
    tolerance_tokens = 10

    pr_results = []
    for tau in tau_values:
        correct = 0
        detected = 0
        total = len(marker_samples)
        errors = []

        for s in marker_samples:
            pred_t = entropy_phase_detection(s["entropies"], window, float(tau))
            gt_t = s["t_star"]
            if pred_t < len(s["entropies"]):
                detected += 1
                error = abs(pred_t - gt_t)
                errors.append(error)
                if error <= tolerance_tokens:
                    correct += 1

        precision = correct / max(detected, 1)
        recall = detected / max(total, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-10)
        mean_error = float(np.mean(errors)) if errors else float('inf')

        pr_results.append({
            "tau": float(tau), "precision": precision, "recall": recall,
            "f1": f1, "detected": detected, "correct": correct,
            "total": total, "mean_error_tokens": mean_error,
        })

    best = max(pr_results, key=lambda x: x["f1"])
    exp2 = {
        "pr_curve": pr_results, "best_tau": best["tau"], "best_f1": best["f1"],
        "best_precision": best["precision"], "best_recall": best["recall"],
        "num_samples": len(marker_samples), "tolerance_tokens": tolerance_tokens,
    }

    print(f"\n  Best τ = {best['tau']:.2f} nats")
    print(f"  F1 = {best['f1']:.3f}  (P={best['precision']:.3f}, R={best['recall']:.3f})")

    tau_05 = next((r for r in pr_results if abs(r["tau"] - 0.5) < 0.01), None)
    if tau_05:
        print(f"  At τ=0.5: F1={tau_05['f1']:.3f}, P={tau_05['precision']:.3f}, R={tau_05['recall']:.3f}")

    save_json(exp2, "experiment2_tau_sweep.json")

    # ========== EXPERIMENT 3 ==========
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: Quantization Distortion")
    print("=" * 60)

    exp3_all = {cfg["name"]: {"kl": [], "compression": []} for cfg in BIT_CONFIGS}
    for d in all_data:
        for cfg in BIT_CONFIGS:
            if cfg["name"] in d["exp3"]:
                exp3_all[cfg["name"]]["kl"].append(d["exp3"][cfg["name"]]["kl"])
                exp3_all[cfg["name"]]["compression"].append(d["exp3"][cfg["name"]]["compression"])

    exp3_configs = []
    print(f"\n{'─' * 70}")
    print(f"  {'Configuration':<25} {'KL Div':>10} {'Compression':>13} {'Eff. Bits':>10}")
    print(f"{'─' * 70}")
    for cfg in BIT_CONFIGS:
        kl_vals = exp3_all[cfg["name"]]["kl"]
        cr_vals = exp3_all[cfg["name"]]["compression"]
        entry = {
            "name": cfg["name"], "b_think": cfg["b_think"], "b_answer": cfg["b_answer"],
            "kl_mean": float(np.mean(kl_vals)) if kl_vals else 0,
            "kl_std": float(np.std(kl_vals)) if kl_vals else 0,
            "compression_mean": float(np.mean(cr_vals)) if cr_vals else 0,
            "compression_std": float(np.std(cr_vals)) if cr_vals else 0,
            "num_samples": len(kl_vals),
        }
        exp3_configs.append(entry)
        eff_bits = 16.0 / entry["compression_mean"] if entry["compression_mean"] > 0 else 0
        print(f"  {entry['name']:<25} {entry['kl_mean']:>8.5f}±{entry['kl_std']:.4f}"
              f" {entry['compression_mean']:>8.1f}×±{entry['compression_std']:.1f}"
              f" {eff_bits:>8.2f}")
    print(f"{'─' * 70}")

    exp3 = {"configs": exp3_configs, "alpha_mean": exp1["alpha_mean"], "alpha_std": exp1["alpha_std"]}
    save_json(exp3, "experiment3_quantization.json")

    # ========== EXPERIMENT 4 ==========
    print("\n" + "=" * 60)
    print("EXPERIMENT 4: GSM8K Accuracy")
    print("=" * 60)

    valid_preds = [d for d in all_data if d["exp4_pred"] is not None]
    correct_count = sum(1 for d in all_data if d["exp4_correct"])
    total_valid = len(valid_preds)
    acc = correct_count / max(total_valid, 1)

    print(f"\n  Baseline (FP16) accuracy: {correct_count}/{total_valid} = {acc:.1%}")
    exp4 = {
        "baseline_accuracy": acc, "baseline_correct": correct_count,
        "baseline_total": total_valid, "num_samples": len(all_data),
    }
    save_json(exp4, "experiment4_gsm8k.json")

    return exp1, exp2, exp3, exp4, all_data


# ============================================================
# PLOTTING (with NaN guards)
# ============================================================

def plot_rho_distribution(all_data):
    rho_think = [d["exp1_agg"]["mean_rho_think"] for d in all_data if d["exp1_agg"]["mean_rho_think"] != 0]
    rho_answer = [d["exp1_agg"]["mean_rho_answer"] for d in all_data if d["exp1_agg"]["mean_rho_answer"] != 0]

    if not rho_think or not rho_answer:
        print("  ⚠️  No valid ρ data to plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(rho_think, bins=20, alpha=0.7, color='#2196F3', label=r'$\rho_{\mathrm{think}}$', edgecolor='white')
    axes[0].hist(rho_answer, bins=20, alpha=0.7, color='#FF9800', label=r'$\rho_{\mathrm{answer}}$', edgecolor='white')
    axes[0].axvline(x=np.mean(rho_think), color='#1565C0', linestyle='--', linewidth=1.5,
                     label=f'Mean think = {np.mean(rho_think):.3f}')
    axes[0].axvline(x=np.mean(rho_answer), color='#E65100', linestyle='--', linewidth=1.5,
                     label=f'Mean answer = {np.mean(rho_answer):.3f}')
    axes[0].set_xlabel('Pairwise Cosine Similarity ρ', fontsize=11)
    axes[0].set_ylabel('Count', fontsize=11)
    axes[0].set_title('Distribution of ρ by Phase', fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=9)

    bp = axes[1].boxplot([rho_think, rho_answer], labels=['Think Phase', 'Answer Phase'],
                          patch_artist=True, widths=0.5)
    bp['boxes'][0].set_facecolor('#BBDEFB')
    bp['boxes'][1].set_facecolor('#FFE0B2')
    axes[1].set_ylabel('Pairwise Cosine Similarity ρ', fontsize=11)
    axes[1].set_title('Think vs Answer Redundancy', fontsize=12, fontweight='bold')
    axes[1].axhspan(0.6, 0.75, alpha=0.15, color='green', label='Paper prediction [0.6, 0.75]')
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(f"{CONFIG['figures_dir']}/fig3_rho_distribution.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(f"{CONFIG['figures_dir']}/fig3_rho_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("  📊 Saved: fig3_rho_distribution.pdf/png")


def plot_tau_precision_recall(exp2):
    if not exp2 or "pr_curve" not in exp2:
        return
    pr = exp2["pr_curve"]
    taus = [r["tau"] for r in pr]
    f1s = [r["f1"] for r in pr]
    precisions = [r["precision"] for r in pr]
    recalls = [r["recall"] for r in pr]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(taus, precisions, 'b-o', markersize=3, label='Precision', linewidth=1.5)
    ax1.plot(taus, recalls, 'r-s', markersize=3, label='Recall', linewidth=1.5)
    ax1.plot(taus, f1s, 'g-^', markersize=3, label='F1', linewidth=1.5)
    ax1.axvline(x=0.5, color='gray', linestyle='--', alpha=0.7, label='τ = 0.5 (paper)')
    best_tau = exp2["best_tau"]
    ax1.axvline(x=best_tau, color='green', linestyle=':', alpha=0.7, label=f'Best τ = {best_tau:.2f}')
    ax1.set_xlabel('Threshold τ (nats)', fontsize=11)
    ax1.set_ylabel('Score', fontsize=11)
    ax1.set_title('Phase Detection: τ Sensitivity', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.set_ylim(-0.05, 1.05)

    ax2.plot(recalls, precisions, 'k-o', markersize=4, linewidth=1.5)
    ax2.set_xlabel('Recall', fontsize=11)
    ax2.set_ylabel('Precision', fontsize=11)
    ax2.set_title('Precision-Recall Curve', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f"{CONFIG['figures_dir']}/fig4_tau_sweep.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(f"{CONFIG['figures_dir']}/fig4_tau_sweep.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("  📊 Saved: fig4_tau_sweep.pdf/png")


def plot_quantization_distortion(exp3):
    if not exp3 or "configs" not in exp3:
        return
    configs = exp3["configs"]
    valid_configs = [c for c in configs if c["kl_mean"] > 0 and not np.isnan(c["kl_mean"])]
    if not valid_configs:
        print("  ⚠️  No valid quantization data to plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {'Uniform': '#E53935', 'TA-TurboQuant': '#1E88E5', 'TA-Conservative': '#43A047'}
    for cfg in valid_configs:
        name = cfg["name"]
        color = colors.get('Uniform' if 'Uniform' in name else ('TA-Conservative' if 'Conservative' in name else 'TA-TurboQuant'), '#333')
        marker = 'o' if 'Uniform' in name else ('D' if 'Conservative' in name else 's')
        ax.errorbar(cfg["compression_mean"], cfg["kl_mean"],
                     xerr=cfg["compression_std"], yerr=cfg["kl_std"],
                     fmt=marker, color=color, markersize=10, capsize=4, linewidth=1.5,
                     label=f'{name} ({cfg["compression_mean"]:.1f}×)')

    ax.set_xlabel('Compression Ratio (vs FP16)', fontsize=12)
    ax.set_ylabel('KL Divergence', fontsize=12)
    ax.set_title('Compression vs Distortion Tradeoff', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='upper left')
    if all(c["kl_mean"] > 0 for c in valid_configs):
        ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(f"{CONFIG['figures_dir']}/fig5_compression_distortion.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(f"{CONFIG['figures_dir']}/fig5_compression_distortion.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("  📊 Saved: fig5_compression_distortion.pdf/png")


def plot_per_layer_rho(all_data):
    layer_rho_think = defaultdict(list)
    layer_rho_answer = defaultdict(list)

    for d in all_data:
        for layer_str, ldata in d["exp1_per_layer"].items():
            layer_idx = int(layer_str)
            if ldata["rho_think"] != 0:
                layer_rho_think[layer_idx].append(ldata["rho_think"])
            if ldata["rho_answer"] != 0:
                layer_rho_answer[layer_idx].append(ldata["rho_answer"])

    if not layer_rho_think:
        print("  ⚠️  No per-layer data to plot.")
        return

    layers = sorted(layer_rho_think.keys())
    think_means = [np.mean(layer_rho_think[l]) for l in layers]
    think_stds = [np.std(layer_rho_think[l]) for l in layers]
    answer_means = [np.mean(layer_rho_answer.get(l, [0])) for l in layers]
    answer_stds = [np.std(layer_rho_answer.get(l, [0])) for l in layers]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.errorbar(layers, think_means, yerr=think_stds, fmt='b-o', markersize=5,
                capsize=3, label=r'$\rho_{\mathrm{think}}$', linewidth=1.5)
    ax.errorbar(layers, answer_means, yerr=answer_stds, fmt='r-s', markersize=5,
                capsize=3, label=r'$\rho_{\mathrm{answer}}$', linewidth=1.5)
    ax.set_xlabel('Layer Index', fontsize=11)
    ax.set_ylabel('Mean Pairwise Cosine Similarity', fontsize=11)
    ax.set_title('Redundancy ρ Across Transformer Layers', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.axhspan(0.6, 0.75, alpha=0.1, color='green', label='Paper range')

    plt.tight_layout()
    plt.savefig(f"{CONFIG['figures_dir']}/fig6_per_layer_rho.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(f"{CONFIG['figures_dir']}/fig6_per_layer_rho.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("  📊 Saved: fig6_per_layer_rho.pdf/png")


# ============================================================
# UTILITIES
# ============================================================

def save_json(data, filename):
    path = os.path.join(CONFIG["output_dir"], filename)
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=convert)
    print(f"  💾 Saved: {path}")


def generate_latex_tables(exp1, exp3):
    latex = []
    if exp1:
        latex.append(r"""
% === TABLE: Measured Redundancy ===
\begin{table}[h]
\centering
\caption{Measured pairwise cosine redundancy $\rho$ on DeepSeek-R1-Distill-Qwen-1.5B.}
\label{tab:measured_rho}
\begin{tabular}{lcc}
\toprule
\textbf{Phase} & \textbf{Mean $\rho$} & \textbf{Std} \\
\midrule
Think  & """ + f"{exp1['rho_think_mean']:.4f}" + r" & " + f"{exp1['rho_think_std']:.4f}" + r""" \\
Answer & """ + f"{exp1['rho_answer_mean']:.4f}" + r" & " + f"{exp1['rho_answer_std']:.4f}" + r""" \\
\midrule
Think fraction $\alpha$ & """ + f"{exp1['alpha_mean']:.3f}" + r" & " + f"{exp1['alpha_std']:.3f}" + r""" \\
\bottomrule
\end{tabular}
\end{table}
""")

    if exp3 and "configs" in exp3:
        latex.append(r"""
% === TABLE: Quantization Comparison ===
\begin{table}[h]
\centering
\caption{Attention distortion and compression for different quantization strategies.}
\label{tab:quant_comparison}
\begin{tabular}{lccccc}
\toprule
\textbf{Configuration} & $b_T$ & $b_A$ & \textbf{KL Div} & \textbf{Compression} \\
\midrule""")
        for cfg in exp3["configs"]:
            latex.append(f"  {cfg['name']} & {cfg['b_think']} & {cfg['b_answer']} & "
                         f"${cfg['kl_mean']:.5f} \\pm {cfg['kl_std']:.4f}$ & "
                         f"${cfg['compression_mean']:.1f}\\times$ \\\\")
        latex.append(r"""\bottomrule
\end{tabular}
\end{table}
""")

    output = "\n".join(latex)
    with open(f"{CONFIG['output_dir']}/latex_tables.tex", 'w') as f:
        f.write(output)
    print(f"  📝 Saved: {CONFIG['output_dir']}/latex_tables.tex")


# ============================================================
# MAIN — Single-pass architecture
# ============================================================

def main():
    print("=" * 60)
    print("  TAQG Paper — Experimental Validation Suite v2")
    print("  Model: DeepSeek-R1-Distill-Qwen-1.5B")
    print("  Dataset: GSM8K")
    print("  Architecture: SINGLE-PASS (generate once, analyze everything)")
    print("=" * 60)

    start_time = time.time()

    model, tokenizer = load_model_and_tokenizer()
    prompts, ground_truths = load_gsm8k_prompts()

    # === SINGLE PASS: generate each prompt once, run ALL experiments ===
    print(f"\n🔄 Processing {len(prompts)} samples (single-pass)...")
    all_data = []
    first_sample_printed = False

    for i, (prompt, gt) in enumerate(tqdm(zip(prompts, ground_truths),
                                          total=len(prompts), desc="Processing")):
        try:
            result = generate_with_kv_extraction(model, tokenizer, prompt)

            # Debug: print first sample details
            if not first_sample_printed:
                text_preview = result["generated_text"][:300]
                print(f"\n  📋 FIRST SAMPLE DEBUG:")
                print(f"     Tokens generated: {result['num_tokens']}")
                print(f"     Detection method: {result['detection_method']}")
                print(f"     t_star: {result['t_star']}")
                print(f"     KV cache layers: {len(result['kv_cache'])}")
                if result['kv_cache']:
                    first_key = list(result['kv_cache'].values())[0][0]
                    print(f"     Key tensor shape: {first_key.shape}")
                    print(f"     Key tensor range: [{first_key.min():.4f}, {first_key.max():.4f}]")
                print(f"     Text preview: {text_preview}...")
                print(f"     Has </think>: {'</think>' in result['generated_text']}")
                first_sample_printed = True

            data = process_sample(result, gt)
            all_data.append(data)

            del result
            gc.collect()
            if CONFIG["device"] == "cuda":
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"\n⚠️  Sample {i} failed: {type(e).__name__}: {repr(e)}")
            if i == 0:
                traceback.print_exc()
            gc.collect()
            try:
                if CONFIG["device"] == "cuda":
                    torch.cuda.empty_cache()
            except Exception:
                pass
            continue

    if not all_data:
        print("❌ No samples processed successfully. Check GPU memory and model.")
        return

    # Aggregate and report all experiments
    exp1, exp2, exp3, exp4, all_data = aggregate_and_report(all_data)

    # Generate plots
    print("\n📊 Generating publication-quality figures...")
    plot_rho_distribution(all_data)
    plot_tau_precision_recall(exp2)
    plot_quantization_distortion(exp3)
    plot_per_layer_rho(all_data)

    # Generate LaTeX tables
    print("\n📝 Generating LaTeX tables...")
    generate_latex_tables(exp1, exp3)

    # Summary
    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"  ALL EXPERIMENTS COMPLETE — {elapsed / 60:.1f} minutes")
    print(f"{'=' * 60}")
    print(f"\n  Output directory: {CONFIG['output_dir']}/")
    for f in sorted(os.listdir(CONFIG["output_dir"])):
        fpath = os.path.join(CONFIG["output_dir"], f)
        if os.path.isfile(fpath):
            size = os.path.getsize(fpath)
            print(f"    {f} ({size / 1024:.1f} KB)")

    print(f"\n  Next steps:")
    print(f"    1. Copy latex_tables.tex content into your paper")
    print(f"    2. Replace [MEASURED] placeholders with the numbers above")
    print(f"    3. Recompile: pdflatex taqg_paper_final.tex (3 times)")
    print(f"    4. Upload final PDF + code zip to Zenodo")


if __name__ == "__main__":
    main()
