#!/usr/bin/env python3
"""
TAQG Diagnostic CLI — Measure per-phase KV cache redundancy.

Usage:
    python run_diagnostic.py --model_id deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
    python run_diagnostic.py --model_id deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --num_samples 20 --quick

Runs on a free Google Colab T4 GPU (16GB VRAM).
"""

import argparse
import gc
import json
import os
import sys
import time
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore", category=UserWarning)


def main():
    parser = argparse.ArgumentParser(
        description="TAQG Diagnostic: Measure per-phase KV cache redundancy"
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        help="HuggingFace model ID",
    )
    parser.add_argument(
        "--num_samples", type=int, default=50, help="Number of GSM8K samples to process"
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=1024, help="Max tokens to generate per sample"
    )
    parser.add_argument(
        "--output_dir", type=str, default="taqg_diagnostic_results", help="Output directory"
    )
    parser.add_argument(
        "--quick", action="store_true", help="Quick mode: 10 samples, fewer layers"
    )
    parser.add_argument(
        "--dtype", type=str, default="float32", choices=["float16", "float32"],
        help="Model dtype (float32 recommended for stability)"
    )
    args = parser.parse_args()

    if args.quick:
        args.num_samples = min(args.num_samples, 10)

    # ── Device setup ──
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 60)
    print("  TAQG Phase Redundancy Diagnostic")
    print("=" * 60)
    print(f"  Device: {device}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"  Model: {args.model_id}")
    print(f"  Samples: {args.num_samples}")
    print()

    # ── Load model ──
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {args.model_id}...")
    dtype = torch.float32 if args.dtype == "float32" else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=dtype, device_map="auto"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Loaded. Parameters: {param_count / 1e6:.0f}M\n")

    # ── Load dataset ──
    from datasets import load_dataset

    print(f"Loading GSM8K ({args.num_samples} samples)...")
    ds = load_dataset("openai/gsm8k", "main", split="test")
    prompts = [item["question"] for item in ds.select(range(args.num_samples))]
    print(f"  Loaded {len(prompts)} prompts.\n")

    # ── Import TAQG modules ──
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from taqg.diagnostic import measure_phase_redundancy, taqg_bit_allocation
    from taqg.utils import generate_with_kv_extraction

    # ── Process samples ──
    from tqdm import tqdm

    all_results = []
    start_time = time.time()

    print("Processing samples...")
    for i, prompt in enumerate(tqdm(prompts, desc="Diagnostic")):
        try:
            result = generate_with_kv_extraction(
                model, tokenizer, prompt,
                max_new_tokens=args.max_new_tokens,
                device=device,
            )

            if not result["kv_cache"]:
                continue

            measurement = measure_phase_redundancy(
                result["kv_cache"], result["t_star"]
            )

            all_results.append({
                "sample_idx": i,
                "rho_think": measurement["rho_think"],
                "rho_answer": measurement["rho_answer"],
                "t_star": result["t_star"],
                "num_tokens": result["num_tokens"],
                "detection_method": result["detection_method"],
                "alpha": result["t_star"] / max(result["num_tokens"], 1),
            })

            # Free memory
            del result
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"\n  Warning: Sample {i} failed: {e}")
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
            continue

    elapsed = time.time() - start_time

    # ── Aggregate results ──
    if not all_results:
        print("\nERROR: No samples completed successfully.")
        return

    rho_think_vals = [r["rho_think"] for r in all_results if r["rho_think"] > 0]
    rho_answer_vals = [r["rho_answer"] for r in all_results if r["rho_answer"] > 0]
    alpha_vals = [r["alpha"] for r in all_results]

    rho_t = float(np.mean(rho_think_vals)) if rho_think_vals else 0.0
    rho_a = float(np.mean(rho_answer_vals)) if rho_answer_vals else 0.0
    rho_t_std = float(np.std(rho_think_vals)) if rho_think_vals else 0.0
    rho_a_std = float(np.std(rho_answer_vals)) if rho_answer_vals else 0.0
    alpha_mean = float(np.mean(alpha_vals))

    # Determine allocation
    if rho_a > rho_t:
        rho_H, rho_L = rho_a, rho_t
        high_phase = "answer"
    else:
        rho_H, rho_L = rho_t, rho_a
        high_phase = "think"

    allocation = taqg_bit_allocation(rho_H, rho_L, b_low=4)
    b_eff = alpha_mean * 4 + (1 - alpha_mean) * allocation["b_high"]
    compression = 16.0 / b_eff if b_eff > 0 else 0

    # ── Print results ──
    print("\n" + "=" * 60)
    print("  TAQG DIAGNOSTIC RESULTS")
    print("=" * 60)
    print(f"\n  Model: {args.model_id}")
    print(f"  Samples: {len(all_results)}/{args.num_samples}")
    print(f"  Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print()
    print(f"  {'─' * 50}")
    print(f"  ρ_think  = {rho_t:.4f} ± {rho_t_std:.4f}")
    print(f"  ρ_answer = {rho_a:.4f} ± {rho_a_std:.4f}")
    print(f"  α (think fraction) = {alpha_mean:.4f}")
    print(f"  {'─' * 50}")
    print()

    if rho_a > rho_t:
        print(f"  ➡ ANSWER phase is MORE redundant (ρ_A > ρ_T)")
        print(f"  ➡ This is a REDUNDANCY REVERSAL vs. full-scale models")
        print(f"  ➡ TAQG recommends: {allocation['b_low']}-bit think, {allocation['b_high']}-bit answer")
    else:
        print(f"  ➡ THINK phase is MORE redundant (ρ_T > ρ_A)")
        print(f"  ➡ This matches expected behavior for large-scale models")
        print(f"  ➡ TAQG recommends: {allocation['b_high']}-bit think, {allocation['b_low']}-bit answer")

    print(f"  ➡ Effective bit rate: {b_eff:.2f}")
    print(f"  ➡ Expected compression: {compression:.1f}×")
    print()

    marker_count = sum(1 for r in all_results if r["detection_method"] == "marker")
    print(f"  Phase detection: {marker_count}/{len(all_results)} via template markers")
    print(f"  {'─' * 50}")

    # ── Save results ──
    os.makedirs(args.output_dir, exist_ok=True)
    output = {
        "model_id": args.model_id,
        "num_samples": len(all_results),
        "rho_think_mean": rho_t,
        "rho_think_std": rho_t_std,
        "rho_answer_mean": rho_a,
        "rho_answer_std": rho_a_std,
        "alpha_mean": alpha_mean,
        "high_redundancy_phase": high_phase,
        "recommended_b_think": allocation["b_low"] if high_phase == "answer" else allocation["b_high"],
        "recommended_b_answer": allocation["b_high"] if high_phase == "answer" else allocation["b_low"],
        "bit_gap": allocation["bit_gap"],
        "b_eff": b_eff,
        "compression": compression,
        "elapsed_seconds": elapsed,
        "per_sample": all_results,
    }

    output_path = os.path.join(args.output_dir, "diagnostic_results.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
