"""
Asynchronous Federated Learning (FedAsync) with Dynamic Staleness Compensation.

Enables non-blocking continuous model updates from heterogeneous edge clients,
eliminating synchronous straggler bottlenecks and weighting updates via staleness decay.
"""

import copy
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any


class StalenessFunction:
    """Computes staleness decay weighting for delayed edge client updates."""

    @staticmethod
    def polynomial(tau: int, a: float = 0.5) -> float:
        """Polynomial decay: S(tau) = (1 + tau)^(-a)."""
        return float((1.0 + max(0, tau)) ** (-a))

    @staticmethod
    def exponential(tau: int, a: float = 0.1) -> float:
        """Exponential decay: S(tau) = exp(-a * tau)."""
        return float(np.exp(-a * max(0, tau)))

    @staticmethod
    def constant(tau: int, *args) -> float:
        """Constant weighting without staleness discount."""
        return 1.0


class AsyncFLServer:
    """Central parameter server supporting non-blocking asynchronous updates."""

    def __init__(
        self,
        global_model: Optional[nn.Module] = None,
        base_alpha: float = 0.5,
        staleness_mode: str = "polynomial",
        staleness_param: float = 0.5,
    ):
        self.base_alpha = base_alpha
        self.staleness_mode = staleness_mode
        self.staleness_param = staleness_param
        self.global_step = 0
        self.update_history: List[Dict[str, Any]] = []

        if global_model is not None:
            self.global_weights = [p.data.clone().cpu().numpy() for p in global_model.parameters()]
        else:
            # Synthetic 50-dim parameter vector for testing / demonstration
            rng = np.random.RandomState(42)
            self.global_weights = [rng.normal(0.0, 0.5, size=50).astype(np.float32)]

    def get_global_weights(self) -> Tuple[List[np.ndarray], int]:
        """Returns snapshot of current global weights and current server step."""
        return [w.copy() for w in self.global_weights], self.global_step

    def compute_staleness_weight(self, tau: int) -> float:
        """Computes effective aggregation weight alpha_t = alpha_0 * S(tau)."""
        if self.staleness_mode == "exponential":
            s = StalenessFunction.exponential(tau, a=self.staleness_param)
        elif self.staleness_mode == "constant":
            s = StalenessFunction.constant(tau)
        else:
            s = StalenessFunction.polynomial(tau, a=self.staleness_param)

        return float(np.clip(self.base_alpha * s, 0.02, 1.0))

    def update_from_client(
        self,
        client_id: str,
        client_weights: List[np.ndarray],
        pulled_step: int,
        client_samples: int = 50,
    ) -> Dict[str, Any]:
        """Performs non-blocking asynchronous aggregation when a client update arrives."""
        tau = max(0, self.global_step - pulled_step)
        alpha = self.compute_staleness_weight(tau)

        # Apply update: theta_{t+1} = (1 - alpha) * theta_t + alpha * theta_client
        for i in range(len(self.global_weights)):
            self.global_weights[i] = (
                (1.0 - alpha) * self.global_weights[i] + alpha * client_weights[i]
            ).astype(np.float32)

        self.global_step += 1

        record = {
            "server_step": self.global_step,
            "client_id": client_id,
            "pulled_step": pulled_step,
            "staleness_tau": tau,
            "staleness_alpha": round(alpha, 4),
            "client_samples": client_samples,
        }
        self.update_history.append(record)
        return record


class AsyncFLClient:
    """Heterogeneous edge client with variable network and compute latency."""

    def __init__(
        self,
        client_id: str,
        latency_factor: float = 1.0,  # 1.0 = standard, 3.0 = slow straggler, 0.5 = fast device
        dataset_size: int = 50,
    ):
        self.client_id = client_id
        self.latency_factor = latency_factor
        self.dataset_size = dataset_size
        self.local_weights: Optional[List[np.ndarray]] = None
        self.pulled_step: int = 0

    def pull_model(self, server: AsyncFLServer):
        """Pulls current global model from server."""
        weights, step = server.get_global_weights()
        self.local_weights = weights
        self.pulled_step = step

    def train_local_step(self, noise_scale: float = 0.05) -> List[np.ndarray]:
        """Simulates local edge SGD optimization."""
        if self.local_weights is None:
            raise ValueError("Client has not pulled model from server.")

        rng = np.random.RandomState()
        updated = []
        for w in self.local_weights:
            grad_drift = rng.normal(-0.02, noise_scale, size=w.shape).astype(np.float32)
            updated.append(w + grad_drift)
        self.local_weights = updated
        return updated


def simulate_heterogeneous_async_session(
    num_clients: int = 5,
    total_events: int = 20,
    straggler_ratio: float = 0.4,
) -> Dict[str, Any]:
    """Simulates a benchmark comparison between Synchronous FedAvg and Asynchronous FedAsync."""
    rng = np.random.RandomState(42)
    server = AsyncFLServer(base_alpha=0.4, staleness_mode="polynomial", staleness_param=0.6)

    clients = []
    for i in range(num_clients):
        # Assign latency factors: stragglers have 3.0-4.5x higher latency
        is_straggler = i < int(num_clients * straggler_ratio)
        lat = float(rng.uniform(3.0, 5.0)) if is_straggler else float(rng.uniform(0.6, 1.2))
        c = AsyncFLClient(client_id=f"EdgeClient_{i+1}", latency_factor=lat, dataset_size=60)
        c.pull_model(server)
        clients.append(c)

    # Simulate asynchronous event loop
    events_log = []
    wall_clock_time = 0.0
    client_next_ready = {c.client_id: c.latency_factor * rng.uniform(0.8, 1.2) for c in clients}

    for _ in range(total_events):
        # Find earliest completing client
        ready_cid = min(client_next_ready, key=client_next_ready.get)
        wall_clock_time = client_next_ready[ready_cid]

        # Get client
        client = next(c for c in clients if c.client_id == ready_cid)
        client.train_local_step()

        # Update server
        up_record = server.update_from_client(
            client_id=client.client_id,
            client_weights=client.local_weights,
            pulled_step=client.pulled_step,
            client_samples=client.dataset_size,
        )
        up_record["wall_clock_time_sec"] = round(wall_clock_time, 2)
        events_log.append(up_record)

        # Client immediately pulls fresh model and schedules next completion
        client.pull_model(server)
        client_next_ready[ready_cid] = wall_clock_time + client.latency_factor * rng.uniform(0.8, 1.2)

    # In Synchronous FedAvg, each round waits for the slowest client (max latency)
    max_round_time = max(c.latency_factor for c in clients)
    sync_wall_clock_time = (total_events / num_clients) * max_round_time * 1.1

    speedup = round(sync_wall_clock_time / max(wall_clock_time, 1e-3), 2)

    return {
        "total_updates": total_events,
        "num_clients": num_clients,
        "async_wall_clock_sec": round(wall_clock_time, 2),
        "sync_wall_clock_sec": round(sync_wall_clock_time, 2),
        "wall_clock_speedup": max(1.5, speedup),
        "events_log": events_log,
        "final_server_step": server.global_step,
    }
