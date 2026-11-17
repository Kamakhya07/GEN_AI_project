import time
import torch
import torch.nn as nn
from typing import Dict, Any, Tuple

def count_parameters(model: nn.Module) -> Dict[str, Any]:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    return {
        "total_parameters": int(total_params),
        "trainable_parameters": int(trainable_params),
        "size_mb": round(float(size_mb), 2)
    }

def benchmark_model_latency(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    device: torch.device,
    warmup_runs: int = 15,
    benchmark_runs: int = 50
) -> Dict[str, float]:
    """
    Measures average inference latency and standard deviation (in milliseconds)
    on the declared hardware with proper GPU synchronization.
    """
    model.eval()
    model.to(device)

    dummy_input = torch.randn(*input_shape, device=device)

    # Warmup runs
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(dummy_input)
            if device.type == "cuda":
                torch.cuda.synchronize()

    # Benchmark runs
    latencies = []
    with torch.no_grad():
        for _ in range(benchmark_runs):
            start = time.perf_counter()
            _ = model(dummy_input)
            if device.type == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()
            latencies.append((end - start) * 1000.0) # in ms

    return {
        "mean_latency_ms": round(float(sum(latencies) / len(latencies)), 2),
        "min_latency_ms": round(float(min(latencies)), 2),
        "max_latency_ms": round(float(max(latencies)), 2)
    }
