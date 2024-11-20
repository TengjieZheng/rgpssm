import torch


def sqrt(x, eps=1e-12):
    """Safe-gradient square root"""
    return torch.sqrt(x + eps)


def squared_scaled_dist(a, b, gamma):
    if a.shape[1] != b.shape[1]:
        raise ValueError("Inconsistent dimensions")

    gamma = gamma.view(1, -1)
    a = a * gamma
    b = b * gamma
    b = b.t()  # final outcome in shape of x row * z col
    a2 = torch.sum(a ** 2, dim=1, keepdim=True)
    b2 = torch.sum(b ** 2, dim=0, keepdim=True)
    ab = a.mm(b)
    d2 = a2 - 2 * ab + b2
    return torch.clamp(d2, min=0)


def scaled_dist(a, b, gamma):
    return sqrt(squared_scaled_dist(a, b, gamma))


def kron(a, b):
    """Kronecker product"""
    ra, ca = a.shape
    rb, cb = b.shape
    return torch.reshape(
        a.reshape(ra, 1, ca, 1) * b.reshape(1, rb, 1, cb), (ra * rb, ca * cb)
    )


def yakron(a, b):
    """Batch Kronecker product"""
    assert a.dim() > 1 and b.dim() > 1
    if a.dim() == 2:
        a = a.unsqueeze(0)
    if b.dim() == 2:
        b = b.unsqueeze(0)

    ba, ra, ca = a.shape
    bb, rb, cb = b.shape

    assert ba == 1 or bb == 1 or ba == bb

    bc = max(ba, bb)

    return torch.reshape(
        a.reshape(ba, ra, 1, ca, 1) * b.reshape(bb, 1, rb, 1, cb), (bc, ra * rb, ca * cb)
    )
