# SYRK

A multi-backend implementation of the Symmetric Rank-K Update (SYRK) operation.

## Overview

SYRK computes the symmetric rank-k update operation:

```
D = α × A × Aᵀ + β × C
```

Where:
- `A` is an (N, K) input matrix
- `C` is an optional (N, N) symmetric matrix
- `α` and `β` are scalar coefficients
- `D` is the (N, N) symmetric output matrix

## Installation

```bash
# Basic installation
pip install .

# With Triton backend support
pip install ".[triton]"

# With all optional dependencies
pip install ".[all]"
```

## Quick Start

```python
import torch
from syrk import syrk

# Create input tensor (bfloat16 on CUDA)
a = torch.randn(1024, 512, dtype=torch.bfloat16, device='cuda')

# Compute A @ A^T
result = syrk(a)
print(result.shape)  # torch.Size([1024, 1024])

# With scaling factors
result = syrk(a, alpha=0.5)

# With additive term
c = torch.randn(1024, 1024, dtype=torch.bfloat16, device='cuda')
c = (c + c.T) / 2  # Make symmetric
result = syrk(a, c=c, alpha=1.0, beta=0.5)
```

## API Reference

### `syrk(a, c=None, alpha=1.0, beta=0.0, skip_upper_triangle=False, backend=None)`

Compute the Symmetric Rank-K Update operation.

**Parameters:**
- `a` (torch.Tensor): Input tensor of shape (N, K). Must be bfloat16.
- `c` (torch.Tensor, optional): Symmetric input tensor of shape (N, N). Required if beta != 0.
- `alpha` (float): Scaling factor for the matrix multiplication. Default: 1.0.
- `beta` (float): Scaling factor for the matrix addition. Default: 0.0.
- `skip_upper_triangle` (bool): If True, only compute the lower triangle. Default: False.
- `backend` (Backend, optional): The backend to use. Auto-selects if None.

**Returns:**
- torch.Tensor: Output tensor of shape (N, N).

## Backends

### Triton Backend

The Triton backend provides a high-performance GPU implementation using NVIDIA's Triton compiler. It requires:
- Triton >= 3.4.0 with TMA (Tensor Memory Accelerator) support
- NVIDIA GPU with appropriate compute capability

```python
from syrk import syrk
from syrk.backends import Backend, get_available_backends

# Check available backends
print(get_available_backends())  # [<Backend.TRITON: 'triton'>]

# Explicitly use Triton backend
result = syrk(a, backend=Backend.TRITON)
```

## Requirements

- Python >= 3.9
- PyTorch >= 2.0.0
- (Optional) Triton >= 3.4.0 for Triton backend

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

The Triton backend implementation is based on code from NVIDIA Corporation, 
originally licensed under the Apache License 2.0.

## Acknowledgments

- NVIDIA Corporation for the original Triton SYRK implementation

