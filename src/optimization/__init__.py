"""
Optimization Module: PyTorch Dynamic Quantization and Latency/Memory Benchmarking.
"""

from .quantize import DynamicQuantizer
from .benchmark import LatencyBenchmark

__all__ = ["DynamicQuantizer", "LatencyBenchmark"]
