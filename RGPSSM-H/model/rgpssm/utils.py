from typing import Optional, Union, Tuple, List

import torch
from numpy import ndarray
from torch import Tensor
from abc import ABC, abstractmethod

import cholup, choldown

class IModel(ABC):

    @abstractmethod
    def fun_input(self, x: Tensor, c: Tensor)-> Tuple[Tensor, Optional[Tensor]]:
        """Get GP input
        Args:
            x: system state (..., nx)
            c: system input (..., nc)
        Returns:
            z: GP input (..., nz)
            dzdx: Jacobin of z w.r.t. x (..., nz, nx) or None
        """

        pass

    @abstractmethod
    def fun_tran(self, x: Tensor, c: Tensor, f: Tensor)-> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
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
    def fun_meas(self, x: Tensor, c: Tensor)-> Tuple[Tensor, Optional[Tensor]]:
        """Measurement model, get measurement
        Args:
            x: system state (..., nx)
            c: system input (..., nc)
        Returns:
            y: measurement (..., ny)
            Cx: measurement Jacobin dy/dx (ny, nx) or None
        """

        pass

def clone_required_grad(x: Optional[Tensor], view:Optional[Union[List, Tuple]]=None)-> Tensor:
    """Clone tensor and set requires_grad = True"""
    if x is None:
        xnew = x
    else:
        xnew = torch.tensor(x.detach().cpu().numpy(), device=x.device, requires_grad=True)
        if view is not None:
            xnew = xnew.view(*view)
    return xnew

def Jacobian(x: Tensor, y: Tensor)->Tensor:
    """Get Jacobian matrix dy/dx based on torch
    Args:
        x : [tensor] need required_grad=True, but don't need used
        y : [tensor] need required_grad=True, but don't need used
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

def ToTensor(x: Union[ndarray, List, Tensor], device=None, dtype=torch.float32, view:Optional[Union[List, Tuple]]=None):
    """to tensor"""
    if x is None:
        y = x
    else:
        if isinstance(x, torch.Tensor):
            y = x
        else:
            y = torch.tensor(x)

        if dtype is not None:
            y = y.to(dtype)

        if device is not None:
            return y.to(device)

    if view is not None and y is not None:
        y = y.view(*view)
    return y


def chol_add(L0, v, m, id):
    """Compute the updated Cholesky factor after adding rows/columns to the original matrix.
    Args:
        L0: original cholesky factor (n, n)
        v: added column (n, d)
        m: added square block (d, d)
        id: adding index
    Return:
        L: new cholesky factor (n+d, n+d)
    Notation:
        original matrix A0 = [A11 A12
                              A21 A22]
        added column v = [v1
                          v2]
        new matrix A = [A11   v1  A12
                        v1.T  m   v2.T
                        A21   v2  A22]
    """

    n, d = v.shape
    assert L0.shape == (n, n), "L0 must be square with shape (n, n)"
    assert m.shape == (d, d), "m must be square with shape (d, d)"
    assert 0 <= id <= n, "id must be between 0 and n"

    if id > 0 and id < n:
        """
        original cholesky factor and matrix: L0 = [A0, 0        L0*L0^T = [A0*A0^T, A0*B0^T
                                                   B0, C0]                 B0*A0^T, B0*B0^T + C0*C0^T]
        new cholesky factor and matrix:      L = [A, 0, 0       L*L^T = [A*A^T A*a^T            A*B^T                   = [A0*A0^T  v1    A0*B0^T
                                                  a, b, 0                a*A^T a*a^T + b*b^T    a*B^T + b*c^T              v1^T     m     v2^T
                                                  B, c, C]               B*A^T B*a^T + c*b^T    B*B^T + c*c^T + C*C^T]     B0*A0^T  v2    B0*B0^T + C0*C0^T]
        where: v1 = v[:id, :], v2 = v[id:, :]
        therefore:  A = A0, a^T = A^-1 * v1, B = B0
                    b = chol(m - a*a^T)
                    c^T = b^-1 * (v2^T - a*B^T)
                    C = chol(C0*C0^T - c*c^T)
        """
        # Partition L0
        A0 = L0[:id, :id]  # (id, id)
        B0 = L0[id:, :id]  # (n - id, id)
        C0 = L0[id:, id:]  # (n - id, n - id)

        v1 = v[:id, :]  # (id, d)
        v2 = v[id:, :]  # (n - id, d)

        A, B = A0, B0
        a = torch.linalg.solve_triangular(A0, v1, upper=False).T  # (id, d)
        b = torch.linalg.cholesky(m - a@a.T)  # (d, d)
        c = torch.linalg.solve_triangular(b, v2.T - a @ B0.T, upper=False).T  # (d, n - id)
        C = choldown.chol_downdate(C0, c)

        L = assemble_chol([[A], [a, b], [B, c, C]])

    elif id == 0:
        """
        new cholesky factor: L = [a, 0  , L*L^T = [a*a^T, a*b^T             = [m, v^T
                                  b, C]            b*a^T, b*b^T + C*C^T]       v, L0*L0^T]
        """
        a = torch.linalg.cholesky(m)
        b = torch.linalg.solve_triangular(a, v.T, upper=False).T
        C = choldown.chol_downdate(L0, b)
        L = assemble_chol([a], [b, C])

    elif id == n:
        """
        new cholesky factor: L = [L0, 0  ,      L*L^T = [L0*L0^T,   L0*rho^T                    = [L0*L0^T, v^T
                                  rho, beta]             rho*L0^T   rho*rho^T + beta*beta^T]       v, m]
        """
        rho = torch.linalg.solve_triangular(L0, v, upper=False).T
        beta = torch.linalg.cholesky(m - rho@rho.T)
        L = assemble_chol([[L0], [rho, beta]])

    return L


def chol_delete(L0, l, r):
    """
    Args:
        L0: original cholesky factor (n, n)
        l:  the left index of the deleted rows and columns
        r: the right index of the deleted rows and columns
    Returns:

    Notation:
        Suppose original Cholesky factor is L0 = [A, 0, 0;  and L0^T = [A^T, a^T, B^T;
                                                  a, b, 0;              0,   b^T, c^T
                                                  B, c, C]              0,   0,   C^T]
        then the original joint covariance is [A*A^T, A*a^T,       A*B^T;
                                               a*A^T, a*a^T+b*b^T, a*B^T+b*c^T
                                               B*A^T, B*a^T+c*b^T, B*B^T+c*c^T+C*C^T]

    """

    n = L0.shape[0]
    if r == n:
        return L0[:l, :l]
    else:
        A, B, C = L0[:l, :l], L0[r:, :l], L0[r:, r:]
        c = L0[r:, l:r]
        C_new = cholup.chol_update(C, c)
        L = assemble_chol([[A], [B, C_new]])
        return L


def assemble_cov(A11, A21, A22):
    """[A11,  A21.T
        A21,  A22   ]"""
    Upper = torch.cat((A11, A21.T), dim=1)
    Lower = torch.cat((A21, A22), dim=1)
    return torch.cat((Upper, Lower), dim=0)

def assemble_chol(L_dict):
    """
    Args:
        L_dict: [[A], [a, b], [B, c, C], ...]
    Returns:
        L
    """

    def get_n_col(l_dict):
        n_cols = [l_dict[i].shape[1] for i in range(len(l_dict))]
        return sum(n_cols)

    def get_raw(L, n_col_last):
        n_col = get_n_col(L)
        mat_zero = torch.zeros((L[0].shape[0], n_col_last - n_col), dtype=L[0].dtype, device=L[0].device)
        return torch.cat((*L, mat_zero), dim=1)

    n_col_last = get_n_col(L_dict[-1])
    return torch.cat([get_raw(L, n_col_last) for L in L_dict], dim=0)

def insert_row(A, r, id):
    """
    Args:
        A: original tensor [n, *]
        r: inserted tensor [m, *]
        id: inserted index
    """

    A1, A2 = A[:id], A[id:]
    return torch.cat((A1, r, A2), dim=0)

def get_mat_left(mat, l, r):
    """Get the left block of a matrix after deleting l-th to r-th rows and coloums"""
    # delete inducing point self-cov

    mu1 = mat[:l, :l]
    mu2 = mat[:l, r:]
    md1 = mat[r:, :l]
    md2 = mat[r:, r:]

    mu = torch.cat([mu1, mu2], dim=1)
    md = torch.cat([md1, md2], dim=1)
    m = torch.cat([mu, md], dim=0)

    return m

def get_vec_left(vec, l, r, dim):
    """Get the left block of a matrix after deleting l-th to r-th rows or coloums"""
    # delete inducing point input / output mean / cov

    if dim == 0:
        v = torch.cat((vec[:l, :], vec[r:, :]), dim=dim)
    elif dim == 1:
        v = torch.cat((vec[:, :l], vec[:, r:]), dim=dim)

    return v

def keep_sym(A):
    """A should theoretically be symmetric, this function aims to eliminate numerical error"""
    return (A + A.T) * 0.5

def nearest_positive_definite(A: torch.Tensor, eps: float = 1e-8):
    """Find the nearest positive definite matrix based on SVD"""
    # Ensure symmetric
    B = (A + A.T) / 2
    # SVD
    U, S, V = torch.svd(B)
    # Truncate negative singular values
    S_pos = torch.clamp(S, min=eps)
    # Reconstruct
    return U @ torch.diag(S_pos) @ U.T

def chol_near_singular(L, tol=1e-12):
    # L: lower-triangular Cholesky factor
    diag = torch.diag(L)
    min_diag = torch.min(diag).item()
    max_diag = torch.max(diag).item()
    cond_est = (max_diag / min_diag) ** 2

    near_singular = min_diag < tol * max_diag
    Lnew = torch.clone(L)
    if near_singular:
        idx_min = torch.argmin(diag)
        Lnew[idx_min, idx_min] = tol * max_diag

    return near_singular, Lnew, cond_est

if __name__ == '__main__':
    x = torch.tensor([1.], requires_grad=True)
    x0 = 1*x
    z = 2*x
    b = torch.tensor([0.], requires_grad=True)
    y = x0 + z

    print(Jacobian(x, y))
    print(Jacobian(x0, y))
    print(Jacobian(b, y))