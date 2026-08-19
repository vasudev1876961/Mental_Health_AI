"""
Personalized Federated Learning Strategy (FedPer).

Splits the neural network into:
1. Shared Global Backbone (aggregated across FL clients).
2. Personalized Local Prediction Heads (retained strictly on client device to adapt to subject-specific baselines).

Usage:
    client = FedPerClient(model, train_loader, personal_head_keys=["risk_head", "classification_head"])
"""

import copy
import torch
import torch.nn as nn
from typing import Dict, List, OrderedDict


class FedPerManager:
    """Manages model state parameter splitting between shared global backbone and local personalized heads."""

    def __init__(self, personal_layer_prefixes: List[str] = None):
        if personal_layer_prefixes is None:
            self.personal_layer_prefixes = ["risk_head", "classification_head", "fatigue_head", "attention_head"]
        else:
            self.personal_layer_prefixes = personal_layer_prefixes

    def is_personal_layer(self, key: str) -> bool:
        """Returns True if the parameter key belongs to a local personalized layer."""
        return any(key.startswith(prefix) for prefix in self.personal_layer_prefixes)

    def extract_global_parameters(self, model_state_dict: OrderedDict) -> OrderedDict:
        """Extracts shared global backbone parameters (filters out local personal heads)."""
        global_params = OrderedDict()
        for k, v in model_state_dict.items():
            if not self.is_personal_layer(k):
                global_params[k] = v
        return global_params

    def merge_global_and_local(
        self, global_params: OrderedDict, local_params: OrderedDict
    ) -> OrderedDict:
        """Merges global shared parameters with local personalized head parameters."""
        merged = copy.deepcopy(local_params)
        for k, v in global_params.items():
            merged[k] = v
        return merged
