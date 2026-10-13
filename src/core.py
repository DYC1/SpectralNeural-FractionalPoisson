import math
import numpy as np
import torch

DOMAINS = ("interval", "disk", "lshape", "square_hole", "annulus")
SEEDS = (20260915, 20260916, 20260917)
S_VALUES = (0.3, 0.5, 0.7)


def geometry(x, domain):
    r2 = x.square().sum(-1)
    if domain in ("interval", "disk"):
        return torch.relu(1 - r2)
    if domain == "annulus":
        return torch.relu(1 - r2) * torch.relu(r2 - 0.04)
    outer = torch.relu(1 - x.square()).prod(-1)
    if domain == "square_hole":
        return outer * torch.relu(r2 - 0.04)
    if domain == "lshape":
        return outer * (
            torch.relu(x[:, 0] * (1.01 - x[:, 0]))
            + torch.relu(x[:, 1] * (1.01 - x[:, 1]))
        )
    raise ValueError(domain)


def boundary_distance(x, domain):
    r = x.square().sum(-1).sqrt()
    if domain in ("interval", "disk"):
        return 1 - r
    if domain == "annulus":
        return torch.minimum(1 - r, r - 0.2)
    outer = (1 - x.abs()).amin(-1)
    if domain == "square_hole":
        return torch.minimum(outer, r - 0.2)
    notch = torch.relu(x).square().sum(-1).sqrt()
    return torch.minimum(outer, notch)


def exact(x, s):
    d = x.shape[-1]
    constant = (
        2 ** (-2 * s) * math.gamma(d / 2) / math.gamma(d / 2 + s) / math.gamma(1 + s)
    )
    return constant * torch.relu(1 - x.square().sum(-1)) ** s


def spectral(N, d, s, device):
    if d not in (1, 2):
        raise ValueError("This attachment supports dimensions 1 and 2 only")
    if N < 2 or not 0 < s < 1:
        raise ValueError("Require N>=2 and 0<s<1")
    dtype = torch.float64
    j = torch.arange(N + 1, device=device, dtype=dtype)
    y = -torch.cos((2 * j + 1) * math.pi / (2 * N + 2))
    x = y / torch.sqrt(1 - y * y)
    w = math.pi / (N + 1) / (1 - y * y)
    T = [torch.ones_like(y), y]
    D = [torch.zeros_like(y), torch.ones_like(y)]
    for k in range(2, N + 1):
        T.append(2 * y * T[-1] - T[-2])
        D.append(4 * y if k == 2 else D[k - 2] * k / (k - 2) + 2 * k * T[k - 1])
    T = torch.stack(T)
    D = torch.stack(D)
    Ts = math.sqrt(2 / math.pi) * T * torch.sqrt(1 - y * y)[None, :]
    Ts[0] /= math.sqrt(2)
    DT = (
        math.sqrt(2 / math.pi)
        * (D * (1 - y * y)[None, :] - T * y[None, :])
        * (1 - y * y)[None, :]
    )
    DT[0] /= math.sqrt(2)
    ev, V = torch.linalg.eigh((DT * w[None, :]) @ DT.T)
    ev = ev.clamp_min(1e-13)
    A = V.T @ Ts
    if d == 1:
        points = x[:, None]
        weights = w
        eig = ev
    else:
        X, Y = torch.meshgrid(x, x, indexing="ij")
        points = torch.stack([X.flatten(), Y.flatten()], 1)
        weights = w[:, None] * w[None, :]
        eig = ev[:, None] + ev[None, :]
    spec = torch.zeros_like(eig)
    for k in range(-100, 101):
        t = k / 3
        spec += math.exp(s * t) * eig / (math.exp(t) + eig)
    scale = 0.25 / 3 * 2 * math.sin(math.pi * s) / math.pi
    return points, weights, A, spec, scale, y, x


class Net(torch.nn.Module):
    def __init__(self, d):
        super().__init__()
        self.InputLayer = torch.nn.Linear(d, 100)
        self.HiddenLayer = torch.nn.ModuleList(
            [torch.nn.Linear(100, 100) for _ in range(2)]
        )
        self.OutputLayer = torch.nn.Linear(100, 1)
        for layer in self.modules():
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_normal_(layer.weight)

    def forward(self, x):
        u = torch.tanh(self.InputLayer(x))
        for layer in self.HiddenLayer:
            u = torch.tanh(layer(u)) + u
        return self.OutputLayer(u).squeeze(-1)


def test_grid(d, device, count=401):
    if d not in (1, 2) or count < 3:
        raise ValueError("Invalid evaluation grid")
    g = torch.linspace(-1, 1, count, device=device, dtype=torch.float64)
    if d == 1:
        return g[:, None]
    X, Y = torch.meshgrid(g, g, indexing="ij")
    return torch.stack([X.flatten(), Y.flatten()], 1)


def interpolate_matrix(y, g):
    N = len(y) - 1
    yt = g / np.sqrt(1 + g * g)
    bw = (-1.0) ** np.arange(N + 1) * np.sin(
        (2 * np.arange(N + 1) + 1) * np.pi / (2 * N + 2)
    )
    diff = yt[:, None] - y[None, :]
    hit = np.abs(diff) < 1e-13
    B = bw[None, :] / np.where(hit, 1, diff)
    B /= B.sum(1)[:, None]
    for idx in np.flatnonzero(hit.any(1)):
        B[idx] = hit[idx].astype(float)
    return B * np.sqrt(1 - yt * yt)[:, None] / np.sqrt(1 - y * y)[None, :]
