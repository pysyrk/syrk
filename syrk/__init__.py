# SPDX-License-Identifier: Apache-2.0
"""SYRK: Multi-backend Symmetric Rank-K Update implementation.

This package provides efficient implementations of the SYRK operation
(Symmetric Rank-K Update) with support for multiple backends.

The SYRK operation computes:
    D = alpha * A @ A^T + beta * C

where A is an (N, K) matrix and C is an optional (N, N) symmetric matrix.

Example:
    >>> import torch
    >>> from syrk import syrk
    >>> a = torch.randn(1024, 512, dtype=torch.bfloat16, device='cuda')
    >>> result = syrk(a)  # Computes a @ a.T
    >>> result.shape
    torch.Size([1024, 1024])

For backend-specific access:
    >>> from syrk.backends import Backend, get_available_backends
    >>> from syrk import syrk
    >>> # Check available backends
    >>> print(get_available_backends())
    >>> # Use specific backend
    >>> result = syrk(a, backend=Backend.TRITON)
"""

__version__ = "0.1.0"

from syrk.backends import Backend, get_available_backends, get_backend
from syrk.core import syrk

__all__ = [
    # Core function
    "syrk",
    # Backend utilities
    "Backend",
    "get_available_backends",
    "get_backend",
    # Version
    "__version__",
]

