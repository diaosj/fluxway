"""Fluxway performance benchmarks.

Modules:
    benchmark_ops      - Core tensor operation benchmarks (argsort, take, bincount, etc.)
    benchmark_gpu      - GPU-specific multi-scenario comparison
    benchmark_dlpack   - DLPack zero-copy interop benchmarks
    benchmark_backends - Multi-backend (JAX/CuPy/Hybrid) full comparison
    benchmark_e2e      - End-to-end communication pipeline benchmarks
"""
