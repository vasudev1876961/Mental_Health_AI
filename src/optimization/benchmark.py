"""
Latency and Inference Benchmarker.

Measures inference latency (ms), FPS, and CPU/GPU memory footprint.
"""

import time
import numpy as np
import torch
import torch.nn as nn
from typing import Dict


class LatencyBenchmark:
    """Measures model inference speed and throughput."""

    def __init__(self):
        pass

    def benchmark_model(
        self,
        model: nn.Module,
        num_warmup: int = 5,
        num_iters: int = 30,
        device: torch.device = torch.device("cpu"),
    ) -> Dict[str, float]:
        """Measures mean inference latency (ms) and Frames-Per-Second (FPS)."""
        model.eval()

        v_seq = torch.randn(1, 30, 18, device=device)
        a_feat = torch.randn(1, 16, device=device)
        t_feat = torch.randn(1, 128, device=device)
        mask = torch.ones(1, 3, device=device)

        # Warmup iterations
        with torch.no_grad():
            for _ in range(num_warmup):
                _ = model(v_seq, a_feat, t_feat, mask=mask)

        # Benchmark iterations
        latencies = []
        with torch.no_grad():
            for _ in range(num_iters):
                start = time.perf_counter()
                _ = model(v_seq, a_feat, t_feat, mask=mask)
                end = time.perf_counter()
                latencies.append((end - start) * 1000.0) # ms

        mean_latency = float(np.mean(latencies))
        p95_latency = float(np.percentile(latencies, 95))
        fps = float(1000.0 / max(mean_latency, 1e-3))

        return {
            "mean_latency_ms": float(np.round(mean_latency, 2)),
            "p95_latency_ms": float(np.round(p95_latency, 2)),
            "fps": float(np.round(fps, 1)),
        }
