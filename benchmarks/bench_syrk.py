#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Benchmark script for SYRK operations."""

import argparse
import time
from typing import Callable

import torch

# Try to import syrk, fall back to direct backend import
try:
    from syrk import syrk, get_available_backends, Backend
    from syrk.backends.triton_backend import HAS_TRITON_TMA
except ImportError:
    syrk = None
    HAS_TRITON_TMA = False

print(f"SYRK: {syrk}")
print(f"HAS_TRITON_TMA: {HAS_TRITON_TMA}")

def benchmark_fn(
    fn: Callable,
    *args,
    warmup_iters: int = 10,
    bench_iters: int = 100,
    **kwargs,
) -> dict:
    """Benchmark a function with warmup.

    Args:
        fn: Function to benchmark.
        *args: Positional arguments to pass to fn.
        warmup_iters: Number of warmup iterations.
        bench_iters: Number of benchmark iterations.
        **kwargs: Keyword arguments to pass to fn.

    Returns:
        Dictionary with timing statistics.
    """
    # Warmup
    for _ in range(warmup_iters):
        fn(*args, **kwargs)
    torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(bench_iters):
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn(*args, **kwargs)
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms

    times = torch.tensor(times)
    return {
        "mean_ms": times.mean().item(),
        "std_ms": times.std().item(),
        "min_ms": times.min().item(),
        "max_ms": times.max().item(),
        "median_ms": times.median().item(),
    }


def torch_syrk(a: torch.Tensor) -> torch.Tensor:
    """Reference SYRK using PyTorch matmul."""
    return a @ a.T


def compute_tflops(n: int, k: int, time_ms: float) -> float:
    """Compute TFLOPS for SYRK operation.
    
    SYRK: C = A @ A^T where A is (N, K)
    FLOPs = 2 * N * N * K (multiply-add for each element)
    """
    flops = 2 * n * n * k
    tflops = flops / (time_ms / 1000) / 1e12
    return tflops


def run_benchmark(
    shapes: list[tuple[int, int]],
    warmup_iters: int = 10,
    bench_iters: int = 100,
    dtype: torch.dtype = torch.bfloat16,
) -> list[dict]:
    """Run benchmarks for given matrix shapes.

    Args:
        shapes: List of (N, K) tuples representing matrix shapes.
        warmup_iters: Number of warmup iterations.
        bench_iters: Number of benchmark iterations.
        dtype: Data type for tensors.

    Returns:
        List of benchmark results.
    """
    results = []

    print("=" * 80)
    print(f"SYRK Benchmark (dtype={dtype}, warmup={warmup_iters}, iters={bench_iters})")
    print("=" * 80)
    print()

    # Check available backends
    if syrk is not None:
        backends = get_available_backends()
        print(f"Available backends: {[b.name for b in backends]}")
    else:
        print("SYRK package not properly installed, using direct import")
    print(f"Triton TMA available: {HAS_TRITON_TMA}")
    print()

    for n, k in shapes:
        print(f"Shape: A[{n}, {k}] -> C[{n}, {n}]")
        print("-" * 60)

        # Create input tensor
        a = torch.randn(n, k, dtype=dtype, device="cuda")

        result = {
            "shape": (n, k),
            "output_shape": (n, n),
        }

        # Benchmark PyTorch reference
        torch_result = benchmark_fn(
            torch_syrk, a,
            warmup_iters=warmup_iters,
            bench_iters=bench_iters,
        )
        result["torch"] = torch_result
        torch_tflops = compute_tflops(n, k, torch_result["mean_ms"])
        print(f"  PyTorch (A @ A.T):  {torch_result['mean_ms']:8.3f} ms "
              f"± {torch_result['std_ms']:.3f} ms  |  {torch_tflops:.2f} TFLOPS")

        # Benchmark Triton SYRK if available
        if HAS_TRITON_TMA and syrk is not None:
            try:
                triton_result = benchmark_fn(
                    syrk, a,
                    warmup_iters=warmup_iters,
                    bench_iters=bench_iters,
                )
                result["triton"] = triton_result
                triton_tflops = compute_tflops(n, k, triton_result["mean_ms"])
                speedup = torch_result["mean_ms"] / triton_result["mean_ms"]
                print(f"  Triton SYRK:        {triton_result['mean_ms']:8.3f} ms "
                      f"± {triton_result['std_ms']:.3f} ms  |  {triton_tflops:.2f} TFLOPS  "
                      f"({speedup:.2f}x vs PyTorch)")
            except Exception as e:
                print(f"  Triton SYRK:        FAILED - {e}")
                result["triton"] = {"error": str(e)}
        else:
            print("  Triton SYRK:        SKIPPED (TMA not available)")
            result["triton"] = {"error": "TMA not available"}

        print()
        results.append(result)

    return results


def print_summary(results: list[dict]) -> None:
    """Print summary table of benchmark results."""
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print()
    print(f"{'Shape':<20} {'PyTorch (ms)':<15} {'Triton (ms)':<15} {'Speedup':<10}")
    print("-" * 60)

    for r in results:
        n, k = r["shape"]
        shape_str = f"[{n}, {k}]"
        
        torch_ms = r["torch"]["mean_ms"]
        
        if "triton" in r and "mean_ms" in r["triton"]:
            triton_ms = r["triton"]["mean_ms"]
            speedup = torch_ms / triton_ms
            print(f"{shape_str:<20} {torch_ms:<15.3f} {triton_ms:<15.3f} {speedup:<10.2f}x")
        else:
            print(f"{shape_str:<20} {torch_ms:<15.3f} {'N/A':<15} {'N/A':<10}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark SYRK operations")
    parser.add_argument(
        "--warmup", type=int, default=10,
        help="Number of warmup iterations (default: 10)"
    )
    parser.add_argument(
        "--iters", type=int, default=100,
        help="Number of benchmark iterations (default: 100)"
    )
    parser.add_argument(
        "--shapes", type=str, default=None,
        help="Custom shapes as 'N1,K1;N2,K2;...' (default: use predefined shapes)"
    )
    args = parser.parse_args()

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. This benchmark requires a GPU.")
        return

    print(f"Device: {torch.cuda.get_device_name(0)}")
    print()

    # Define shapes to benchmark
    if args.shapes:
        shapes = []
        for shape_str in args.shapes.split(";"):
            n, k = map(int, shape_str.split(","))
            shapes.append((n, k))
    else:
        # Default shapes from user request
        shapes = [
            (128, 2048),
            (2048, 2048),
            (2048, 6144),
            (6144, 2048),
        ]

    # Run benchmarks
    results = run_benchmark(
        shapes=shapes,
        warmup_iters=args.warmup,
        bench_iters=args.iters,
    )

    # Print summary
    print_summary(results)


if __name__ == "__main__":
    main()

