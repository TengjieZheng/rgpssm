from typing import Optional, Union, Tuple, List
import copy

import torch
import numpy as np
from numpy import ndarray
from torch import Tensor
from abc import ABC, abstractmethod

class IModelEKF(ABC):

    @abstractmethod
    def fun_tran(self, x: Tensor, c: Tensor)-> Tuple[Tensor, Optional[Tensor]]:
        """Transition model, get next system state
        Args:
            x: system state (..., nx)
            c: system input (..., nc)
        Returns:
            F: next system state (..., nx)
            Ax: Jacobin for system state dF/dx (..., nx, nx) or None
        """

        pass

    @abstractmethod
    def fun_meas(self, x: Tensor, c: Tensor)-> Tuple[Tensor, Optional[Tensor]]:
        """Measurement model, get measurement
        Args:
            x: system state (..., nx)
            c: system input (..., nc)
        Returns:
            y: measurement (..., ny)
            Cx: measurement Jacobin dy/dx (..., ny, nx) or None
        """

        pass


def clone_required_grad(x):
    xnew = torch.tensor(x.detach().cpu().numpy(), device=x.device, requires_grad=True)
    return xnew

def Jacobian(x: Tensor, y: Tensor)->Tensor:
    """Get Jacobian matrix dy/dx based on torch
    Args:
        x : [tensor]
        y : [tensor]
    Returns:
        J : Jacobian matrxi [tensor]
    """

    y = y.view(-1)
    J = torch.zeros((y.numel(), x.numel())).to(x.device)
    for ii in range(y.numel()):
        yy = y[ii]
        dyy_dx = torch.autograd.grad(yy, x, create_graph=True, retain_graph=True, allow_unused=True)[0]
        if dyy_dx is None: dyy_dx = torch.zeros_like(x)
        J[ii, :] = dyy_dx.view(1, -1)

    return J

def Np2Torch(x: ndarray, device=None):
    """ndarray to tensor"""
    if x is None:
        return x
    else:
        if isinstance(x, torch.Tensor):
            y = x.to(torch.float32)
        else:
            y = torch.tensor(x, dtype=torch.float32)

        if device is not None:
            return y.to(device)

    return y