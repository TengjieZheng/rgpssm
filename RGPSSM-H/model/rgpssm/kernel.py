from typing import Optional, Union, Tuple, List
from numpy import ndarray
from torch import Tensor
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import gpytorch
import torch.nn.functional as F

def inv_softplus(x):
    return x + torch.log(-torch.expm1(-x))

def softplus(x):
    return F.softplus(x)


class IKer(ABC, nn.Module):
    def __init__(self, df=1, dz=1, jitter=0):
        super().__init__()
        self.df = df
        self.dz = dz
        self.jitter = jitter

    @ abstractmethod
    def forward(self, Z1: Tensor, Z2: Optional[Tensor]=None)-> Tensor:
        """
        Args:
            Z1: each row is a GP input (*, N1, dz)
            Z2: each row is a GP input (*, N2, dz)
        Returns:
            K: i-th, j-th block of each batch is the GP prior variance of f(z1^i, z2^j), (*, N1*df, N2*df)
        """
        pass

    @property
    @abstractmethod
    def ls(self) -> Tensor:
        pass

    @property
    @abstractmethod
    def var(self) -> Tensor:
        pass



class MultiTaskRBFKer(IKer):
    def __init__(self, df: int, dz: int, std: Union[List, ndarray, Tensor], ls: Union[List, ndarray, Tensor], jitter=1e-4):
        super().__init__(df, dz, jitter)
        self.ls_raw = nn.Parameter(inv_softplus(torch.tensor(ls)).view(-1))
        self.var_raw = nn.Parameter(inv_softplus(torch.tensor(std) ** 2).view(-1))

    def forward(self, Z1: Tensor, Z2: Optional[Tensor]=None) -> Tensor:
        """
        Args:
            Z1: each row is a GP input (*, N1, dz)
            Z2: each row is a GP input (*, N2, dz)
        Returns:
            K: i-th, j-th block of each batch is the GP prior variance of f(z1^i, z2^j), (*, N1*df, N2*df)
        """

        flag_add_jitter = False
        if Z2 is None:
            Z2 = Z1
            flag_add_jitter = True
        else:
            if torch.equal(Z1, Z2):
                flag_add_jitter = True

        Z1_ = Z1 / self.ls.view(-1)                     # (*, N1, dz)
        Z2_ = Z2 / self.ls.view(-1)                     # (*, N2, dz)

        d = Z1_.unsqueeze(-2) - Z2_.unsqueeze(-3)       # (*, N1, N2, dz)
        tmp = torch.sum(d ** 2, dim=-1)                 # (*, N1, N2)
        cov_x = torch.exp(-0.5 * tmp)                   # (*, N1, N2)
        cov_y = torch.diag_embed(self.var.view(-1))     # (df, df)

        if flag_add_jitter:
            Jitter = torch.eye(cov_x.shape[-1], device=cov_x.device) * self.jitter  # (N1, N1)
        else:
            Jitter = 0.
        K = torch.kron(cov_x + Jitter, cov_y)

        return K

    @property
    def ls(self) -> Tensor:
        return softplus(self.ls_raw).view(1, -1)

    @property
    def var(self) -> Tensor:
        return softplus(self.var_raw).view(1, -1)

    def dK_dz1(self, Z1: Tensor, Z2: Optional[Tensor]=None) ->Tensor:
        """
        Args:
            Z1: (*, N1, dz)
            Z2: (*, N2, dz)
        Returns:
            dKdz1: (*, dz, N1*df, N2*df)
        kernel:
                K(x1, x2) = v * exp[-0.5 * (x1 - x2)^T @ Lam^-1 @ (x1 - x2)]
                where Lam = diag(lam) and lam = [l1^2, ..., ln^2]
        Then,
            dK(x1, x2)/dx1      = -K(x1,x2) * Lam^-1 @ (x1 - x2)
            d^2K(x1, x2)/dx1dx2 = K(x1,x2) * (Lam^-1 - Lam^-1 (x1 - x2) (x1 - x2)^T Lam^-1)
        """

        if Z2 is None: Z2 = Z1
        d = -(Z1.unsqueeze(-2) - Z2.unsqueeze(-3))                      # (*, N1, N2, dz)
        invlam_d = d * self.ls.view(-1)**-2                             # (*, N1, N2, dz)
        invlam_d = invlam_d.permute(*range(Z1.dim()-2), -1, -3, -2)     # (*, dz, N1, N2)
        K = self.forward(Z1, Z2)                                        # (*, N1*df, N2*df)

        a_expd = invlam_d.repeat_interleave(self.df, dim=-2).repeat_interleave(self.df, dim=-1)      # (*, dz, N1*df, N2*df)
        K_expd = K.unsqueeze(-3).expand(*Z1.shape[:-2], self.dz, -1, -1)                             # (*, dz, N1*df, N2*df)
        dKdz1 = a_expd * K_expd                                                                      # (*, dz, N1*df, N2*df)

        return dKdz1

    def dK_dz1dz2(self, Z1: Tensor, Z2: Optional[Tensor]=None) ->Tensor:
        """
        Args:
            Z1: (*, N1, dz)
            Z2: (*, N2, dz)
        Returns:
            dKdz1: (*, dz, dz, N1*df, N2*df)
        """

        if Z2 is None: Z2 = Z1
        d = Z1.unsqueeze(-2) - Z2.unsqueeze(-3)                                         # (*, N1, N2, dz)
        d_Lam = d * self.ls.view(-1)**-2                                                # (*, N1, N2, dz)
        tmp = torch.einsum('...i, ...j->...ij', d_Lam, d_Lam)                     # (*, N1, N2, dz, dz)
        invLam = torch.diag_embed(self.ls.view(-1) ** -2)                               # (dz, dz)
        tmp = invLam - tmp                                                              # (*, N1, N2, dz, dz)
        tmp = tmp.permute(*range(Z1.dim()-2), -2, -1, -4, -3)                           # (*, dz, dz， N1, N2)
        tmp = tmp.repeat_interleave(self.df, dim=-2).repeat_interleave(self.df, dim=-1) # (*, dz, dz， N1*df, N2*df)

        K = self.forward(Z1, Z2)                                                        # (*, N1*df, N2*df)
        K = K.unsqueeze(-3).unsqueeze(-3)                                               # (*, 1, 1, N1*df, N2*df)
        dKdz1dz2 = K * tmp                                                              # (*, dz, dz， N1*df, N2*df)

        return dKdz1dz2



class MultiTaskRBFKerGpytotch(MultiTaskRBFKer):
    def __init__(self, df: int, dz: int, std: Union[List, ndarray, Tensor], ls: Union[List, ndarray, Tensor], jitter=1e-5):
        super().__init__(df, dz, std, ls, jitter)
        del self.ls_raw
        del self.var_raw

        self.ker = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=dz), num_tasks=df, rank=0) # Kxx \otimes V, V = diag([\sigma_i ^ 2])
        self.ker.task_covar_module.var = (torch.tensor(std).view(-1, 1)) ** 2
        self.ker.data_covar_module.lengthscale = torch.tensor(ls).view(-1, 1)

    def forward(self, Z1: Tensor, Z2: Optional[Tensor]=None) -> Tensor:
        if Z2 is None:
            K = self.ker(Z1).to_dense()
            K = self.add_jitter(K)
        else:
            K = self.ker(Z1, Z2).to_dense()
            if torch.equal(Z1, Z2):
                K = self.add_jitter(K)

        return K

    @property
    def ls(self) -> Tensor:
        return self.ker.data_covar_module.lengthscale.view(1, -1)

    @property
    def var(self) -> Tensor:
        return self.ker.task_covar_module.var.view(1, -1)

    def add_jitter(self, K, jitter=None):
        """add positive values proportional to output std for every input"""

        if jitter is None:
            jitter = self.jitter

        var = self.var.detach().clone().view(-1)
        num_input = int(K.shape[0] / self.df)
        Jitter = torch.kron(torch.eye(num_input), torch.diag_embed(var) * jitter)
        Knew = K + Jitter

        return Knew

if __name__ == '__main__':
    torch.manual_seed(1)
    ker = MultiTaskRBFKer(df=1, dz=1, std=[0.1], ls=[0.1])
    Z1 = torch.tensor([0.1, 0.2, 0.3]).view(-1, 1)
    Z2 = torch.tensor([0.12, 0.26]).view(-1, 1)
    K = ker(Z1, Z2)
    dKdz1 = ker.dK_dz1(Z1, Z2)
    dKdz1dz2 = ker.dK_dz1dz2(Z1, Z2)

    print(dKdz1)
    print(dKdz1dz2)

