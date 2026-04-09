"""
TAQG: Type-Aware Quantization Gap for KV Cache Compression
============================================================

A framework for phase-aware KV cache quantization in reasoning LLMs.
Proves uniform quantization is suboptimal when think-phase and answer-phase
tokens differ in pairwise cosine redundancy.

Paper: https://doi.org/10.5281/zenodo.19482477
"""

__version__ = "2.0.0"
__author__ = "Raviteja Nekkalapu"

from taqg.diagnostic import measure_phase_redundancy, taqg_bit_allocation
from taqg.quantize import simulate_phase_aware_quantization
from taqg.utils import extract_kv_cache, find_think_boundary
