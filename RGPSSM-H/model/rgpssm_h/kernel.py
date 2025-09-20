from typing import Optional, Union, Tuple, List
from numpy import ndarray
from torch import Tensor
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import gpytorch
import torch.nn.functional as F
from torch.xpu import device

from ..rgpssm.utils import ToTensor
from ..rgpssm.kernel import inv_softplus, softplus


class IKerH(ABC, nn.Module):
    def __init__(self, df: Union[List, Tensor], dz: Union[List, Tensor], jitter=1e-4):
        super().__init__()
        self.df = ToTensor(df, dtype=torch.int, view=(-1,))
        self.dz = ToTensor(dz, dtype=torch.int, view=(-1,))
        self.df_max = self.df.max()
        self.dz_max = self.dz.max()
        self.nf = self.df.sum()

        self.id_z = torch.arange(0, self.df.numel(), dtype=torch.int)   # index for a single Z [0, 1, ..., nf]
        self.id_f = self.idZ_to_idf(self.id_z)                          # index for a single f [0, 1, 1, ..., nf, nf, nf]

        self.jitter = jitter

    @ abstractmethod
    def forward(self, Z1: Tensor, Id1: Tensor, Z2: Optional[Tensor]=None, Id2: Optional[Tensor]=None, flag_sort=False)-> Tensor:
        """
        Args:
            Z1: N1 elements and each element is a GP input (N1, dz_max)
            Id1: each element represent the id of a dimension of f (N1,)
            Z2: N2 elements and each element is a GP input (N2, dz_max)
            Id2: each element represent the id of a dimension of f (N1,)
            flag_sort: whether input sorted inducing inputs
        Returns:
            K: i-th, j-th block is the GP prior covariance between f(z1^i; id1^i) and f(z2^j; id2^j), (N1*df, N2*df)
        """
        pass

    @property
    @abstractmethod
    def ls(self)-> List[Tensor]:
        """
        Returns: [(1, dz_i)]
        """
        pass

    @property
    @abstractmethod
    def var(self)-> List[Tensor]:
        """
        Returns: [(1, df_i)]
        """
        pass

    def z_extract(self, z: Tensor, id: int)-> Tensor:
        """Extract z by squeeze the additional zeros
        Args:
            z: (*, dz_max)
            id: int
        Returns:
            (*, dz_id)
        """
        return z[..., :self.dz[id]]

    def z_fill(self, z: Union[Tensor, List[Tensor]])-> Tensor:
        """Pad zeros to make the dimension of z equal to dz_ma
        Args:
           z[Tensor]: (*, dz) -> (*, dz_max)
           z[List[Tensor]]: [(*, dz)] -> [(*, dz_max)]
        """
        if isinstance(z, Tensor):
            return self._z_fill_tensor(z)
        elif isinstance(z, list):
            z_ = [self._z_fill_tensor(zi) for zi in z]
            return torch.cat(z_, dim=-2)

    def _z_fill_tensor(self, z:Tensor)-> Tensor:
        """
        Args：
            z: (*, dz)
        Returns:
            (*, dz_max)
        """
        tmp = torch.zeros(*z.shape[:-1], self.dz_max-z.shape[-1], device=z.device, requires_grad=z.requires_grad)
        return torch.cat((z, tmp), dim=-1)

    def idZ_to_idf(self, id_Z):
        id_f = []
        for ii in id_Z.view(-1):
            id_f = id_f + [ii] * self.df[ii]
        id_f = torch.tensor(id_f)
        return id_f


class RBFKerH(IKerH):
    def __init__(self, df: List, dz: List, std: List[List[float]], ls: List[List[float]], jitter=1e-4):
        super().__init__(df, dz, jitter)

        self.ls_raw = nn.ParameterList([nn.Parameter(inv_softplus(ToTensor(l)).view(-1)) for l in ls])          # [tensor]
        self.var_raw = nn.ParameterList([nn.Parameter(inv_softplus(ToTensor(s) ** 2).view(-1)) for s in std])   # [tensor]

    def forward(self, Z1: Tensor, Id1: Tensor, Z2: Optional[Tensor]=None, Id2: Optional[Tensor]=None, flag_sort=False) -> Tensor:
        """
        Args:
            Z1: N1 elements and each element is a GP input (B, N1, dz_max)
            Id1: each element represent the id of a dimension of f (N1,)
            Z2: N2 elements and each element is a GP input (B, N2, dz_max)
            Id2: each element represent the id of a dimension of f (N1,)
            flag_sort: whether input sorted inducing inputs
        Returns:
            K: i-th, j-th block is the GP prior covariance between f(z1^i; id1^i) and f(z2^j; id2^j), (B, N1*df, N2*df)
        """

        if Z2 is None or Id2 is None:
            Z2 = Z1
            Id2 = Id1

        K = self._K_sort(Z1, Id1, Z2, Id2) if flag_sort else self._K(Z1, Id1, Z2, Id2)

        return K

    def _K(self, Z1, Id1, Z2, Id2):
        """Evaluate the kernel matrix when Z are sorted by function output
        Args:
            Z1: N1 elements and each element is a GP input (B, N1, dz_max)
            Id1: each element represent the id of a dimension of f (N1,)
            Z2: N2 elements and each element is a GP input (B, N2, dz_max)
            Id2: each element represent the id of a dimension of f (N1,)
        Returns:
            K: (B, N1*df, N2*df)
        """

        flag_batch = Z1.dim() == 3
        B = Z1.shape[0] if flag_batch else 1

        # Sort
        sort1 = torch.sort(Id1) # (N1)
        sort2 = torch.sort(Id2) # (N2)

        # Evaluate kernel matrix
        K_blocks = []
        for i in range(len(self.df)):
            id_selected1 = sort1.indices[sort1.values == i] # (n1)
            id_selected2 = sort2.indices[sort2.values == i] # (n2)
            n1, n2 = len(id_selected1), len(id_selected2)
            if n1 != 0 and n2 != 0:
                z1 = self.z_extract(Z1[:, id_selected1], i) if flag_batch else self.z_extract(Z1[id_selected1], i)
                z2 = self.z_extract(Z2[:, id_selected2], i) if flag_batch else self.z_extract(Z2[id_selected2], i)
                K_blocks.append(self.ker_single(z1, z2, i))
            else:
                if flag_batch:
                    K_zero = torch.zeros(B, n1 * self.df[i], n2 * self.df[i]).to(Z1[0].device)
                else:
                    K_zero = torch.zeros(n1 * self.df[i], n2 * self.df[i]).to(Z1[0].device)
                K_blocks.append(K_zero)
        K_sort = self.blk_diag(K_blocks, flag_batch, B)

        # Recover
        Nf1 = self.df[sort1.values]
        Nf2 = self.df[sort2.values]
        id_sort1 = torch.repeat_interleave(sort1.indices, Nf1)
        id_sort2 = torch.repeat_interleave(sort2.indices, Nf2)

        id_recover1 = torch.argsort(id_sort1)
        id_recover2 = torch.argsort(id_sort2)

        if flag_batch:
            K = K_sort[:, id_recover1, :][:, :, id_recover2]
        else:
            K = K_sort[id_recover1, :][:, id_recover2]

        return K

    def _K_sort(self, Z1, Id1, Z2, Id2):
        """Evaluate the kernel matrix when Z are sorted by function output
        Args:
            Z1: N1 elements and each element is a GP input (B, N1, dz_max)
            Id1: each element represent the id of a dimension of f (N1)
            Z2: N2 elements and each element is a GP input (B, N2, dz_max)
            Id2: each element represent the id of a dimension of f (N1)
        Returns:
            K: (B, N1*df, N2*df)
        """

        flag_batch = Z1.dim() == 3
        B = Z1.shape[0] if flag_batch else 1

        K_blocks = []
        for i in range(len(self.df)):
            id_selected1 = Id1 == i                             # (N1)
            id_selected2 = Id2 == i                             # (N2)
            n1, n2 = id_selected1.sum(), id_selected2.sum()
            if n1 !=0 and n2 !=0:
                z1 = self.z_extract(Z1[:, id_selected1], i) if flag_batch else self.z_extract(Z1[id_selected1], i) # (B, N1, dz_i)
                z2 = self.z_extract(Z2[:, id_selected2], i) if flag_batch else self.z_extract(Z2[id_selected2], i) # (B, N2, dz_i)
                K_blocks.append(self.ker_single(z1, z2, i))     # (B, N1*dfi, N2*dfi)
            else:
                if flag_batch:
                    K_zero = torch.zeros(B, n1 * self.df[i], n2 * self.df[i]).to(Z1[0].device)
                else:
                    K_zero = torch.zeros(n1*self.df[i], n2*self.df[i]).to(Z1[0].device)
                K_blocks.append(K_zero)

        K = self.blk_diag(K_blocks, flag_batch, B)


        return K

    def blk_diag(self, K_blocks, flag_batch=False, B=1):
        """
        Args:
            K_blocks: [(ni, ni)] or [(B, ni, ni)]
            flag_batch: whether K_blocks has batch dimension
            B: batch size
        Returns:
            K: (N, N) or (B, N, N)
        """
        if flag_batch:
            n = len(K_blocks)
            K_batch = [torch.block_diag(*[K_blocks[j][i] for j in range(n)]).unsqueeze(0)
                       for i in range(B)]
            K = torch.cat(K_batch, dim=0)
        else:
            K = torch.block_diag(*K_blocks)

        return K

    @property
    def ls(self) -> List[Tensor]:
        """
        Returns: [(1, dz_i)]
        """
        return [F.softplus(l).view(1, -1) for l in self.ls_raw]
        

    @property
    def var(self) -> List[Tensor]:
        """
        Returns: [(1, df_i)]
        """
        return [softplus(v).view(1, -1) for v in self.var_raw]
    
    def ker_single(self, Z1: Tensor, Z2: Tensor, i: int) -> Tensor:
        """
        Args:
            Z1: each row is a GP input (*, N1, dz)
            Z2: each row is a GP input (*, N2, dz)
            i: index of the dimension of f
        Returns:
            K: i-th, j-th block is the GP prior variance of f(z1^i, z2^j), (*, N1*df, N2*df)
        """

        ls = self.ls[i]
        var = self.var[i]

        if torch.equal(Z1, Z2):
            flag_add_jitter = True
        else:
            flag_add_jitter = False

        Z1_ = Z1 / ls.view(-1) # (*, N1, dz)
        Z2_ = Z2 / ls.view(-1) # (*, N2, dz)

        d = Z1_.unsqueeze(-2) - Z2_.unsqueeze(-3) # (*, N1, N2, dz)
        tmp = torch.sum(d**2, dim=-1)             # (*, N1, N2)
        cov_x = torch.exp(-0.5 * tmp)             # (*, N1, N2)
        cov_y = torch.diag_embed(var.view(-1))    # (df, df)

        if flag_add_jitter:
            Jitter = torch.eye(cov_x.shape[-1]).to(Z1_.device) * self.jitter  # (N1, N1)
        else:
            Jitter = 0.
        K = torch.kron(cov_x + Jitter, cov_y)   # (*, N1*df, N2*df)

        return K

    def ker_single_base(self, Z1: Tensor, Z2: Tensor, Lambda: Tensor):
        """ exp(-0.5 * (Z1 - Z2)^T Lambda^{-1} (Z1 - Z2))
        Args:
            Z1: each row is a GP input (*, N1, dz)
            Z2: each row is a GP input (N2, dz)
            Lambda: covariance (dz,) or (dz, dz)
        Returns:
            K: i-th, j-th block is the GP prior variance of f(z1^i, z2^j), (*, N1, N2)
        """

        if Lambda.dim() == 1:
            ls = Lambda**0.5
            Z1_ = Z1 / ls.view(-1)                                          # (* N1, dz)
            Z2_ = Z2 / ls.view(-1)                                          # (N2, dz)
        elif Lambda.dim() == 2:
            L = torch.linalg.cholesky(Lambda)
            Z1_ = torch.linalg.solve_triangular(L, Z1.transpose(-1, -2), upper=False).transpose(-1, -2)     # (* N1, dz)
            Z2_ = torch.linalg.solve_triangular(L, Z2.transpose(-1, -2), upper=False).transpose(-1, -2)     # (N2, dz)
        else:
            raise ValueError('Lambda must be 1- or 2-dimensional.')

        d = Z1_.unsqueeze(-2) - Z2_.unsqueeze(-3)  # (*, N1, N2, dz)
        tmp = torch.sum(d ** 2, dim=-1)
        K = torch.exp(-0.5 * tmp)

        return K


class LinearRBFKerHi(RBFKerH):
    def __init__(self, df: List, dz: List, std: List[List[float]], ls: List[List[float]], std_lin: List[List[float]], ls_lin: List[List[float]], jitter=1e-4):
        super().__init__(df, dz, std, ls, jitter)

        self.ls_lin_raw = nn.ParameterList([nn.Parameter(inv_softplus(torch.tensor(l)).view(-1)) for l in ls_lin])
        self.var_lin_raw = nn.ParameterList([nn.Parameter(inv_softplus(torch.tensor(s) ** 2).view(-1)) for s in std_lin])

    @property
    def ls_lin(self) -> List[Tensor]:
        """
        Returns: [(1, dz_i)]
        """
        return [F.softplus(l).view(1, -1) for l in self.ls_lin_raw]

    @property
    def var_lin(self) -> List[Tensor]:
        """
        Returns: [(1, df_i)]
        """
        return [softplus(v).view(1, -1) for v in self.var_lin_raw]


    def ker_single(self, Z1: Tensor, Z2: Tensor, i: int) -> Tensor:
        """
        Args:
            Z1: each row is a GP input (N1, dz)
            Z2: each row is a GP input (N2, dz)
            i: index of the dimension of f
        Returns:
            K: i-th, j-th block is the GP prior variance of f(z1^i, z2^j), (N1*df, N2*df)
        """

        ls, ls_lin = self.ls[i], self.ls_lin[i]
        var, var_lin = self.var[i], self.var_lin[i]

        if torch.equal(Z1, Z2):
            flag_add_jitter = True
        else:
            flag_add_jitter = False

        Z1_ = Z1 / ls.view(-1)  # (N1, dz)
        Z2_ = Z2 / ls.view(-1)  # (N2, dz)

        d = Z1_.unsqueeze(1) - Z2_.unsqueeze(0)  # (N1, N2, dz)
        tmp = torch.sum(d ** 2, dim=-1)
        cov_x = torch.exp(-0.5 * tmp)
        cov_y = torch.diag_embed(var.view(-1))

        if flag_add_jitter:
            Jitter = torch.eye(cov_x.shape[0]) * self.jitter
        else:
            Jitter = 0.
        K = torch.kron(cov_x + Jitter, cov_y)

        Z1_lin = Z1 / ls_lin.view(-1)  # (N1, dz)
        Z2_lin = Z2 / ls_lin.view(-1)  # (N2, dz)
        cov_x_lin = Z1_lin @ Z2_lin.T
        cov_y_lin  = torch.diag_embed(var_lin.view(-1))
        K_lin = torch.kron(cov_x_lin, cov_y_lin)

        return K + K_lin


if __name__ == '__main__':
    df = [2, 3]
    dz = [1, 2]
    std = [[0.1] * 2, [0.2] * 3]
    ls = [[0.1] * 1, [0.2] * 2]

    ker = RBFKerH(df, dz, std, ls)

    Id1 = torch.tensor([0, 1, 0])
    Id2 = torch.tensor([1, 0])
    Z1 = [torch.tensor([[0.1, 0.]]),
          torch.tensor([[0.1, 0.2]]),
          torch.tensor([[0.2, 0.]])]
    Z2 = [torch.tensor([[0.1, 0.05]]),
          torch.tensor([[0.05, 0.]])]

    Z1 = torch.cat(Z1, dim=0)
    Z2 = torch.cat(Z2, dim=0)
    K = ker(Z1, Id1, Z2, Id2)
    print(K)

    Z1_ = Z1.unsqueeze(0).expand(2, -1, -1)
    Z2_ = Z2.unsqueeze(0).expand(2, -1, -1)
    K_ = ker(Z1_, Id1, Z2_, Id2)
    print(K_)
