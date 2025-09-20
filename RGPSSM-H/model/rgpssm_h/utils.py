from typing import Optional, Union, Tuple, List

import torch
from numpy import ndarray
from torch import Tensor
from abc import ABC, abstractmethod


class IModel_H(ABC):

    @abstractmethod
    def fun_input(self, x: Tensor, c: Tensor)-> Tuple[Tensor, Optional[List[Tensor]]]:
        """Get GP input
        Args:
            x: system state (..., dx)
            c: system input (..., dc)
        Returns:
            z: GP input (..., nf, dz_max)
            dzdx: Jacobin of flattened z w.r.t. x with shape (..., dz_max*nf, nx) or None
        """
        pass

    @abstractmethod
    def fun_tran(self, x: Tensor, c: Tensor, f:Tensor)-> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        """Transition model, get next system state
        Args:
            x: system state (..., nx)
            c: system input (..., nc)
            f: GP output (..., nf)
        Returns:
            F: next system state (..., nx)
            Ax: Jacobin for system state dF/dx (nx, nx) or None
            Af: Jacobin for GP output dF/df (nx, nf) or None
        """
        pass

    @abstractmethod
    def fun_meas(self, x: Tensor, c: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        """Measurement model, get measurement
        Args:
            x: system state (..., nx)
            c: system input (..., nc)
        Returns:
            y: measurement (..., ny)
            Cx: measurement Jacobin dy/dx (ny, nx) or None
        """

        pass

def extract_diag_blocks(A: torch.Tensor, block_size: int) -> torch.Tensor:
    """
    Extracts square diagonal blocks from batched square matrices.

    Args:
        A: Tensor of shape (..., N, N), a batch of square matrices.
        block_size: Integer M, the size of each square diagonal block.
                    Must divide N evenly.

    Returns:
        Tensor of shape (..., num_blocks, M, M), where num_blocks = N // M,
        containing the diagonal blocks from each matrix in the batch.
    """
    *batch_shape, N, N_ = A.shape
    assert N == N_, "Input must be square matrices"
    assert N % block_size == 0, "block_size must divide matrix size N"

    num_blocks = N // block_size

    # Reshape into (..., num_blocks, block_size, num_blocks, block_size)
    A_reshaped = A.reshape(*batch_shape, num_blocks, block_size, num_blocks, block_size)

    # Extract diagonal blocks (i == j) using .diagonal
    diag_blocks = A_reshaped.diagonal(dim1=-4, dim2=-2)  # (..., num_blocks, block_size, block_size)
    return diag_blocks