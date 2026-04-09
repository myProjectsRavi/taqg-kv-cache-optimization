"""
TAQG Quantization Simulation.

Simulates phase-aware KV cache quantization by applying uniform
scalar quantization at per-phase bit widths, then measuring the
KL divergence between original and quantized attention distributions.
"""

import numpy as np
import torch
import torch.nn.functional as F


def uniform_quantize(tensor, bits):
    """Apply uniform scalar quantization to a tensor.

    Maps values to a grid of 2^bits levels within the tensor's
    dynamic range, then dequantizes back to floating point.

    Args:
        tensor: input tensor to quantize.
        bits: int, number of quantization bits.

    Returns:
        Dequantized tensor (same shape, approximate values).
    """
    if bits >= 16:
        return tensor.clone()

    num_levels = 2 ** bits
    t_min = tensor.min()
    t_max = tensor.max()
    scale = (t_max - t_min) / (num_levels - 1)

    if scale < 1e-10:
        return tensor.clone()

    quantized = torch.round((tensor - t_min) / scale)
    quantized = quantized.clamp(0, num_levels - 1)
    dequantized = quantized * scale + t_min
    return dequantized


def compute_attention_kl(original_keys, quantized_keys, query_vectors):
    """Compute KL divergence between original and quantized attention.

    Uses actual answer-phase key vectors as proxy queries (not random
    vectors), since answer-phase queries are the primary consumers of
    think-phase KV entries during generation.

    Args:
        original_keys: tensor (seq_len, head_dim), original key vectors.
        quantized_keys: tensor (seq_len, head_dim), quantized key vectors.
        query_vectors: tensor (n_queries, head_dim), proxy query vectors.

    Returns:
        float, mean KL divergence across queries.
    """
    d = original_keys.shape[-1]
    scale = 1.0 / (d ** 0.5)

    # Compute attention distributions
    orig_scores = (query_vectors @ original_keys.T) * scale
    quant_scores = (query_vectors @ quantized_keys.T) * scale

    orig_probs = F.softmax(orig_scores, dim=-1).clamp(min=1e-10)
    quant_probs = F.softmax(quant_scores, dim=-1).clamp(min=1e-10)

    # KL(original || quantized)
    kl = (orig_probs * (orig_probs.log() - quant_probs.log())).sum(dim=-1)
    return kl.mean().item()


def simulate_phase_aware_quantization(kv_cache, t_star, b_think, b_answer,
                                       layers_to_sample=None):
    """Simulate phase-aware quantization and measure distortion.

    Quantizes think-phase keys at b_think bits and answer-phase keys
    at b_answer bits, then computes KL divergence using answer-phase
    keys as proxy queries.

    Args:
        kv_cache: dict mapping layer_idx -> (keys, values) tensors.
        t_star: int, phase boundary token index.
        b_think: int, bit width for think-phase tokens.
        b_answer: int, bit width for answer-phase tokens.
        layers_to_sample: optional list of layer indices.

    Returns:
        dict with kl_divergence, compression_ratio, and per-layer details.
    """
    if layers_to_sample is None:
        layers_to_sample = list(kv_cache.keys())

    kl_values = []
    for layer_idx in layers_to_sample:
        keys, _ = kv_cache[layer_idx]
        k = keys[0].mean(dim=0).float()  # (seq_len, head_dim)

        if k.shape[0] < 4 or t_star < 2 or t_star >= k.shape[0] - 1:
            continue

        # Split phases
        k_think = k[:t_star]
        k_answer = k[t_star:]

        # Quantize each phase at its assigned bit width
        k_think_q = uniform_quantize(k_think, b_think)
        k_answer_q = uniform_quantize(k_answer, b_answer)

        # Reconstruct full quantized key sequence
        k_quantized = torch.cat([k_think_q, k_answer_q], dim=0)

        # Use answer-phase keys as query proxies
        queries = k_answer[:min(10, k_answer.shape[0])]
        if queries.shape[0] < 1:
            continue

        kl = compute_attention_kl(k, k_quantized, queries)
        if not (np.isnan(kl) or np.isinf(kl)):
            kl_values.append(kl)

    # Compute effective compression
    alpha = t_star / (t_star + (k.shape[0] - t_star)) if t_star > 0 else 0.5
    b_eff = alpha * b_think + (1 - alpha) * b_answer
    compression = 16.0 / b_eff if b_eff > 0 else 0.0

    return {
        "kl_divergence": float(np.mean(kl_values)) if kl_values else 0.0,
        "kl_std": float(np.std(kl_values)) if kl_values else 0.0,
        "compression_ratio": compression,
        "b_eff": b_eff,
        "b_think": b_think,
        "b_answer": b_answer,
        "alpha": alpha,
        "num_layers": len(kl_values),
    }
