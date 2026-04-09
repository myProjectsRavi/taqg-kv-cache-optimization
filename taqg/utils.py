"""
TAQG Utilities: KV cache extraction and phase boundary detection.

Handles the complexities of extracting KV caches from different
transformers versions (DynamicCache, legacy tuples, etc.) and
detecting think/answer phase boundaries via template markers
or entropy-based fallback.
"""

import numpy as np
import torch


def extract_kv_cache(past_key_values, input_len):
    """Extract KV cache tensors, handling all transformers cache formats.

    Supports DynamicCache (public/private attributes), legacy tuple
    format, and the to_legacy_cache() conversion method.

    Args:
        past_key_values: model output past_key_values (any format).
        input_len: int, number of input (prompt) tokens to skip.

    Returns:
        dict mapping layer_idx -> (keys, values) CPU tensors,
        containing only generated tokens (prompt tokens excluded).
    """
    kv_cache = {}
    if past_key_values is None:
        return kv_cache

    # Approach 1: DynamicCache with public attributes
    if hasattr(past_key_values, "key_cache") and hasattr(
        past_key_values, "value_cache"
    ):
        keys_list = past_key_values.key_cache
        vals_list = past_key_values.value_cache
        for layer_idx in range(len(keys_list)):
            k, v = keys_list[layer_idx], vals_list[layer_idx]
            kv_cache[layer_idx] = (
                k[:, :, input_len:, :].cpu(),
                v[:, :, input_len:, :].cpu(),
            )
        return kv_cache

    # Approach 2: DynamicCache with private attributes
    if hasattr(past_key_values, "_key_cache") and hasattr(
        past_key_values, "_value_cache"
    ):
        keys_list = past_key_values._key_cache
        vals_list = past_key_values._value_cache
        for layer_idx in range(len(keys_list)):
            k, v = keys_list[layer_idx], vals_list[layer_idx]
            kv_cache[layer_idx] = (
                k[:, :, input_len:, :].cpu(),
                v[:, :, input_len:, :].cpu(),
            )
        return kv_cache

    # Approach 3: to_legacy_cache() method
    if hasattr(past_key_values, "to_legacy_cache"):
        legacy = past_key_values.to_legacy_cache()
        for layer_idx, (k, v) in enumerate(legacy):
            kv_cache[layer_idx] = (
                k[:, :, input_len:, :].cpu(),
                v[:, :, input_len:, :].cpu(),
            )
        return kv_cache

    # Approach 4: Direct iteration (legacy tuple format)
    try:
        for layer_idx, layer_data in enumerate(past_key_values):
            if isinstance(layer_data, (tuple, list)) and len(layer_data) >= 2:
                k, v = layer_data[0], layer_data[1]
            elif isinstance(layer_data, torch.Tensor):
                continue
            else:
                continue
            kv_cache[layer_idx] = (
                k[:, :, input_len:, :].cpu(),
                v[:, :, input_len:, :].cpu(),
            )
        if kv_cache:
            return kv_cache
    except Exception:
        pass

    print(f"  Warning: Unrecognized KV cache type: {type(past_key_values).__name__}")
    return kv_cache


def find_think_boundary(generated_text, token_ids, tokenizer):
    """Find the think/answer phase boundary in generated text.

    Primary method: search for </think> template marker.
    Fallback: use a heuristic based on fraction of total tokens.

    Args:
        generated_text: str, the decoded generated text.
        token_ids: list of int, generated token IDs.
        tokenizer: the model's tokenizer.

    Returns:
        tuple of (t_star, method) where t_star is the boundary token
        index and method is "marker" or "heuristic".
    """
    # Method 1: Template marker detection
    markers = ["</think>", "<|end_think|>", "[/THINK]"]
    for marker in markers:
        idx = generated_text.find(marker)
        if idx != -1:
            prefix = generated_text[:idx + len(marker)]
            prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
            t_star = min(len(prefix_tokens), len(token_ids) - 1)
            return t_star, "marker"

    # Fallback: heuristic (assume 75% think, 25% answer)
    t_star = int(len(token_ids) * 0.75)
    return t_star, "heuristic"


def entropy_phase_detection(entropies, window_size=20, tau=0.5):
    """Detect phase boundary using sliding-window entropy drop.

    Scans the per-token entropy sequence for a drop exceeding tau
    within a sliding window, indicating the transition from high-entropy
    think phase to low-entropy answer phase.

    Note: This method performs poorly on small distilled models (F1~0.2).
    Template markers are strongly preferred when available.

    Args:
        entropies: list of float, per-token entropy values.
        window_size: int, number of tokens in the sliding window.
        tau: float, entropy drop threshold in nats.

    Returns:
        int, predicted boundary token index. Returns len(entropies)
        if no boundary is detected.
    """
    if len(entropies) < window_size * 2:
        return len(entropies)

    for t in range(window_size, len(entropies) - window_size):
        left_mean = np.mean(entropies[t - window_size : t])
        right_mean = np.mean(entropies[t : t + window_size])
        if left_mean - right_mean > tau:
            return t

    return len(entropies)


def generate_with_kv_extraction(model, tokenizer, prompt, max_new_tokens=1024,
                                 device="cuda"):
    """Generate text and extract KV cache + per-token entropy.

    Uses the model's chat template for proper tokenization, samples
    with temperature=0.6 and top_p=0.95 (greedy decoding produces
    degenerate output on distilled models), and runs a forward pass
    to extract the full KV cache.

    Args:
        model: the loaded HuggingFace model.
        tokenizer: the model's tokenizer.
        prompt: str, the user prompt.
        max_new_tokens: int, maximum tokens to generate.
        device: str, device to run on.

    Returns:
        dict with generated_text, token_ids, kv_cache, token_entropies,
        t_star, detection_method, num_tokens, and input_len.
    """
    import torch.nn.functional as Fn

    # Use chat template for proper tokenization
    try:
        messages = [{"role": "user", "content": prompt}]
        token_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        )
        input_ids = torch.tensor([token_ids], dtype=torch.long).to(device)
    except Exception:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]

    input_len = input_ids.shape[1]
    attention_mask = torch.ones_like(input_ids)

    # Generate with sampling (float32 for stability)
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

    # Per-token entropy
    token_entropies = []
    for score in outputs.scores:
        probs = Fn.softmax(score[0].float(), dim=-1).clamp(min=1e-10)
        entropy = -(probs * probs.log()).sum().item()
        if np.isnan(entropy) or np.isinf(entropy):
            entropy = 0.0
        token_entropies.append(entropy)

    # Forward pass for KV cache
    full_ids = outputs.sequences[:, : input_len + len(generated_ids)]
    with torch.no_grad():
        kv_outputs = model(input_ids=full_ids, use_cache=True)

    kv_cache = extract_kv_cache(kv_outputs.past_key_values, input_len)
    t_star, method = find_think_boundary(
        generated_text, generated_ids.tolist(), tokenizer
    )

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
