"""
TAQG Diagnostic: Measure per-phase KV cache redundancy.

This module provides the core measurement functions for computing
pairwise cosine redundancy (rho) separately for think-phase and
answer-phase tokens. These measurements drive the TAQG bit-allocation
theorem.
"""

import math
import numpy as np
import torch
import torch.nn.functional as F


def compute_pairwise_cosine_redundancy(kv_cache, t_star, layers_to_sample=None):
    """Compute mean pairwise cosine similarity for think vs answer phase.

    For each sampled layer, extracts key vectors, splits them at the
    phase boundary t_star, normalizes, and computes the mean of the
    upper-triangular pairwise cosine similarity matrix for each phase.

    Args:
        kv_cache: dict mapping layer_idx -> (keys, values) tensors.
            Keys shape: (batch, heads, seq_len, head_dim)
        t_star: int, token index of the think/answer boundary.
        layers_to_sample: optional list of layer indices to analyze.
            If None, uses all layers in kv_cache.

    Returns:
        dict with per-layer results and aggregated rho_think, rho_answer.
    """
    per_layer_results = {}
    all_rho_think = []
    all_rho_answer = []

    if layers_to_sample is None:
        layers_to_sample = list(kv_cache.keys())

    for layer_idx in layers_to_sample:
        keys, _ = kv_cache[layer_idx]
        # Average across attention heads: (seq_len, head_dim)
        k = keys[0].mean(dim=0).float()

        if k.shape[0] < 4 or t_star < 2:
            continue

        # Skip zero-norm vectors to avoid NaN in cosine similarity
        norms = k.norm(dim=-1)
        valid = norms > 1e-8
        if valid.sum() < 4:
            continue

        k_normed = F.normalize(k, dim=-1)

        # Split at phase boundary
        k_think = k_normed[:t_star]
        k_answer = k_normed[t_star:]

        rho_think = _mean_pairwise_cosine(k_think)
        rho_answer = _mean_pairwise_cosine(k_answer)

        if rho_think is not None:
            all_rho_think.append(rho_think)
        if rho_answer is not None:
            all_rho_answer.append(rho_answer)

        per_layer_results[layer_idx] = {
            "rho_think": rho_think,
            "rho_answer": rho_answer,
            "n_think": k_think.shape[0],
            "n_answer": k_answer.shape[0],
        }

    return {
        "per_layer": per_layer_results,
        "mean_rho_think": float(np.mean(all_rho_think)) if all_rho_think else 0.0,
        "mean_rho_answer": float(np.mean(all_rho_answer)) if all_rho_answer else 0.0,
        "std_rho_think": float(np.std(all_rho_think)) if all_rho_think else 0.0,
        "std_rho_answer": float(np.std(all_rho_answer)) if all_rho_answer else 0.0,
        "num_layers_sampled": len(all_rho_think),
    }


def _mean_pairwise_cosine(vectors):
    """Compute mean pairwise cosine similarity for a set of L2-normalized vectors.

    Args:
        vectors: tensor of shape (n, d), assumed L2-normalized.

    Returns:
        float or None if fewer than 2 vectors.
    """
    n = vectors.shape[0]
    if n < 2:
        return None

    # Compute upper-triangular pairwise similarities
    sim_matrix = vectors @ vectors.T
    upper_tri = sim_matrix.triu(diagonal=1)
    n_pairs = n * (n - 1) / 2
    mean_sim = upper_tri.sum().item() / n_pairs
    return mean_sim


def measure_phase_redundancy(kv_cache, phase_boundary, layers_to_sample=None):
    """High-level convenience function for phase redundancy measurement.

    This is the primary user-facing API. It wraps
    compute_pairwise_cosine_redundancy and returns a clean summary.

    Args:
        kv_cache: dict mapping layer_idx -> (keys, values) tensors.
        phase_boundary: int, token index where think -> answer transition occurs.
        layers_to_sample: optional list of layer indices.

    Returns:
        dict with rho_think, rho_answer, recommended allocation, and details.
    """
    result = compute_pairwise_cosine_redundancy(
        kv_cache, phase_boundary, layers_to_sample
    )

    rho_t = result["mean_rho_think"]
    rho_a = result["mean_rho_answer"]

    # Determine which phase is more redundant
    if rho_a > rho_t:
        high_rho_phase = "answer"
        rho_H, rho_L = rho_a, rho_t
    else:
        high_rho_phase = "think"
        rho_H, rho_L = rho_t, rho_a

    # Compute TAQG bit allocation
    allocation = taqg_bit_allocation(rho_H, rho_L, b_low=4)

    return {
        "rho_think": rho_t,
        "rho_answer": rho_a,
        "std_rho_think": result["std_rho_think"],
        "std_rho_answer": result["std_rho_answer"],
        "high_redundancy_phase": high_rho_phase,
        "recommended_b_think": allocation["b_think"],
        "recommended_b_answer": allocation["b_answer"],
        "bit_gap": allocation["bit_gap"],
        "expected_b_eff": allocation["b_eff"],
        "per_layer": result["per_layer"],
        "num_layers_sampled": result["num_layers_sampled"],
    }


def taqg_bit_allocation(rho_H, rho_L, b_low=4):
    """Apply the TAQG theorem to compute optimal per-phase bit widths.

    Implements Theorem 1 from the paper:
        b_H <= b_L - floor(log2((1 - rho_L) / (1 - rho_H) + 1))

    Args:
        rho_H: float, pairwise cosine redundancy of the high-redundancy phase.
        rho_L: float, pairwise cosine redundancy of the low-redundancy phase.
        b_low: int, bit width for the low-redundancy phase (default 4).

    Returns:
        dict with b_high, b_low, bit_gap, and effective bit rate.
    """
    if rho_H <= rho_L:
        # No gap: both phases have same redundancy
        return {
            "b_high": b_low,
            "b_low": b_low,
            "b_think": b_low,
            "b_answer": b_low,
            "bit_gap": 0,
            "b_eff": float(b_low),
        }

    ratio = (1 - rho_L) / (1 - rho_H)
    bit_gap = int(math.floor(math.log2(ratio + 1)))
    b_high = max(b_low - bit_gap, 1)  # Minimum 1 bit

    return {
        "b_high": b_high,
        "b_low": b_low,
        "b_think": b_low if rho_H != rho_L else b_low,
        "b_answer": b_high if rho_H != rho_L else b_low,
        "bit_gap": bit_gap,
        "b_eff": None,  # Needs alpha to compute
    }
