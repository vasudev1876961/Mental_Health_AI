"""
PyTorch Model Dynamic Quantization Module.

Applies INT8 quantization to linear layers for low-latency edge deployment.
"""

import os
import torch
import torch.nn as nn
from typing import Tuple


class DynamicQuantizer:
    """Quantizes PyTorch models to INT8 precision."""

    def __init__(self):
        pass

    def quantize(self, model: nn.Module) -> nn.Module:
        """Applies dynamic quantization to Linear layers."""
        model.eval()
        try:
            quantized_model = torch.ao.quantization.quantize_dynamic(
                model, {nn.Linear}, dtype=torch.qint8
            )
            return quantized_model
        except Exception:
            # Fallback if quantization API varies
            return model

    def get_model_size_mb(self, model: nn.Module, save_path: str = "temp_model.pt") -> float:
        """Measures disk footprint of model checkpoint in Megabytes."""
        torch.save(model.state_dict(), save_path)
        size_bytes = os.path.getsize(save_path)
        if os.path.exists(save_path):
            os.remove(save_path)
        return float(np.round(size_bytes / (1024 * 1024), 2))
