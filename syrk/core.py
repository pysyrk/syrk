# SPDX-License-Identifier: Apache-2.0
"""Core SYRK operations with unified interface."""

from typing import Optional

import torch

from syrk.backends import Backend, get_backend

__all__ = ["syrk", "Backend"]


def syrk(
    a: torch.Tensor,
    c: Optional[torch.Tensor] = None,
    alpha: float = 1.0,
    beta: float = 0.0,
    skip_upper_triangle: bool = False,
    backend: Optional[Backend] = None,
) -> torch.Tensor:
    """Compute the Symmetric Rank-K Update (SYRK) operation.

    Computes: D = alpha * A @ A^T + beta * C

    This operation produces a symmetric matrix D from input matrix A.
    If beta != 0, then C must be a symmetric matrix.

    Args:
        a: Input tensor of shape (N, K). Must be bfloat16.
        c: Optional symmetric input tensor of shape (N, N). Required if beta != 0.
        alpha: Scaling factor for the matrix multiplication. Default: 1.0.
        beta: Scaling factor for the matrix addition. Default: 0.0.
        skip_upper_triangle: If True, only compute and store the lower triangle.
            Default: False.
        backend: The backend to use for computation. If None, automatically
            selects the best available backend.

    Returns:
        Output tensor of shape (N, N) containing the symmetric result.

    Raises:
        TypeError: If input tensor is not bfloat16 or not 2D.
        RuntimeError: If tensor shapes are incompatible or backend is unavailable.

    Example:
        >>> import torch
        >>> from syrk import syrk
        >>> a = torch.randn(1024, 512, dtype=torch.bfloat16, device='cuda')
        >>> result = syrk(a)  # Computes a @ a.T
        >>> result.shape
        torch.Size([1024, 1024])
    """
    backend_module = get_backend(backend)
    return backend_module.tsyrk_ex(
        a=a,
        c=c,
        alpha=alpha,
        beta=beta,
        skip_upper_triangle=skip_upper_triangle,
    )

