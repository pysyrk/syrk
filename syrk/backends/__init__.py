# SPDX-License-Identifier: Apache-2.0
"""SYRK backends module.

This module provides different backend implementations for the SYRK operation.
Currently supported backends:
- triton: High-performance Triton-based implementation (NVIDIA)
"""

from enum import Enum
from typing import Optional

__all__ = ["Backend", "get_available_backends", "get_backend"]


class Backend(Enum):
    """Enumeration of available SYRK backends."""

    TRITON = "triton"
    # Future backends can be added here:
    # CUBLAS = "cublas"
    # TORCH = "torch"


def get_available_backends() -> list[Backend]:
    """Get a list of available backends.

    Returns:
        List of available Backend enum values.
    """
    available = []

    # Check Triton backend
    try:
        from syrk.backends.triton_backend import is_available as triton_available

        if triton_available():
            available.append(Backend.TRITON)
    except ImportError:
        pass

    return available


def get_backend(backend: Optional[Backend] = None):
    """Get a backend module.

    Args:
        backend: The backend to use. If None, automatically selects the best available.

    Returns:
        The backend module.

    Raises:
        RuntimeError: If the requested backend is not available.
    """
    available = get_available_backends()

    if backend is None:
        # Auto-select: prefer Triton if available
        if Backend.TRITON in available:
            backend = Backend.TRITON
        elif available:
            backend = available[0]
        else:
            raise RuntimeError(
                "No SYRK backend is available. "
                "Please install triton>=3.4.0 for Triton backend support."
            )

    if backend == Backend.TRITON:
        if Backend.TRITON not in available:
            raise RuntimeError(
                "Triton backend is not available. "
                "Please install triton>=3.4.0 with TMA support."
            )
        from syrk.backends import triton_backend

        return triton_backend

    raise ValueError(f"Unknown backend: {backend}")

