# TAQG: Type-Aware KV Cache Quantization for Reasoning LLMs

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19482477.svg)](https://doi.org/10.5281/zenodo.19482477)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

**We prove that uniform KV cache quantization is provably suboptimal for reasoning models.** The more redundant generation phase tolerates fewer bits — and which phase that is depends on model size.

> On DeepSeek-R1-Distill-1.5B, answer tokens are MORE redundant than think tokens (ρ_A=0.544 > ρ_T=0.463). Following the TAQG theorem cuts attention distortion by **58%** vs uniform 3-bit quantization.

---

## Key Result

| Configuration | Think bits | Answer bits | KL Divergence | Compression |
|---|---|---|---|---|
| Uniform 4-bit | 4 | 4 | 0.00056 ± 0.0001 | 4.0× |
| Uniform 3-bit | 3 | 3 | 0.00303 ± 0.0004 | 5.3× |
| Uniform 2-bit | 2 | 2 | 0.02008 ± 0.0037 | 8.0× |
| **TAQG-aligned (4/3)** | **4** | **3** | **0.00126 ± 0.0005** | **4.3×** |
| TAQG-aligned (4/2) | 4 | 2 | 0.01348 ± 0.0046 | 4.6× |
| Anti-aligned (3/4) | 3 | 4 | 0.00234 ± 0.0006 | 4.9× |

The theory-prescribed allocation (4/3) achieves **58% lower distortion** than uniform 3-bit and nearly **half the distortion** of the anti-aligned (wrong direction) configuration.

## The Redundancy Reversal

On the full 671B DeepSeek-R1, think-phase tokens are more redundant (R-KV, 2025). But on the 1.5B distilled variant, **answer-phase tokens are more redundant**. Distillation compresses reasoning into diverse representations while answers stay templated.

**If you blindly follow big-model guidance on a distilled model, you compress the wrong phase.**

```
671B model:  ρ_think >> ρ_answer  →  compress think  →  (2/4) allocation
1.5B model:  ρ_answer > ρ_think  →  compress answer  →  (4/3) allocation  ← REVERSED
```

## Quick Start

### Install

```bash
git clone https://github.com/myProjectsRavi/taqg-kv-cache-optimization.git
cd taqg-kv-cache-optimization
pip install -r requirements.txt
```

### Measure Your Model's Phase Redundancy

```bash
python scripts/run_diagnostic.py \
    --model_id deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --num_samples 50
```

Output:
```
TAQG DIAGNOSTIC RESULTS
──────────────────────────────────────────────────
  ρ_think  = 0.4627 ± 0.0404
  ρ_answer = 0.5440 ± 0.0451
  α (think fraction) = 0.7550

  ➡ ANSWER phase is MORE redundant (ρ_A > ρ_T)
  ➡ This is a REDUNDANCY REVERSAL vs. full-scale models
  ➡ TAQG recommends: 4-bit think, 3-bit answer
  ➡ Expected compression: 4.3×
──────────────────────────────────────────────────
```

### Run Full Experiment Suite

```bash
python scripts/run_full_experiment.py
```

This produces all four experiments from the paper:
- **Experiment 1**: Pairwise cosine redundancy (ρ) measurement
- **Experiment 2**: Entropy-based phase detection validation
- **Experiment 3**: Quantization distortion analysis (6 configurations)
- **Experiment 4**: GSM8K accuracy baseline

## The TAQG Theorem

For two generation phases with pairwise cosine redundancies ρ_H > ρ_L ≥ 0:

```
b_H ≤ b_L - ⌊log₂((1-ρ_L)/(1-ρ_H) + 1)⌋
```

The more redundant phase (H) **always** requires strictly fewer bits than the less redundant phase (L). This bound is:
- **Direction-agnostic**: works regardless of which phase is more redundant
- **Closed-form**: no iterative optimization needed
- **Training-free**: just measure ρ and apply

### Python Implementation

```python
from taqg import taqg_bit_allocation

# Measured on DeepSeek-R1-Distill-1.5B
allocation = taqg_bit_allocation(rho_H=0.544, rho_L=0.463, b_low=4)
print(f"High-ρ phase: {allocation['b_high']} bits")  # → 3
print(f"Low-ρ phase: {allocation['b_low']} bits")     # → 4
```

## Repository Structure

```
taqg-kv-cache-optimization/
├── taqg/                          # Core library
│   ├── diagnostic.py              # Redundancy measurement & TAQG theorem
│   ├── quantize.py                # Phase-aware quantization simulation
│   └── utils.py                   # KV cache extraction & phase detection
├── scripts/
│   ├── run_diagnostic.py          # CLI diagnostic tool
│   └── run_full_experiment.py     # Full experiment suite (paper replication)
├── results/                       # Pre-computed results from the paper
├── paper/
│   └── taqg_paper_final.pdf       # Full paper
└── requirements.txt
```

## Hardware Requirements

| Task | GPU | Time |
|---|---|---|
| Diagnostic (50 samples) | Free Colab T4 (16GB) | ~1 hour |
| Full experiments (50 samples) | T4 or better | ~1.5 hours |
| Diagnostic (10 samples, `--quick`) | T4 | ~15 min |

**No A100 required.** Everything runs on free Google Colab.

## Citation

```bibtex
@misc{nekkalapu2026taqg,
  title={Think Less, Store Smarter: A Theoretical Framework for Type-Aware
         KV Cache Quantization in Large Reasoning Models},
  author={Nekkalapu, Raviteja},
  year={2026},
  doi={10.5281/zenodo.19482477},
  url={https://doi.org/10.5281/zenodo.19482477}
}
```

## Paper

📄 [Read the full paper (PDF)](paper/taqg_paper_final.pdf) | 🔗 [Zenodo Record](https://doi.org/10.5281/zenodo.19482477)

## License

MIT License. See [LICENSE](LICENSE) for details.
