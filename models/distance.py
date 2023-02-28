import torch
from torch import nn
import numpy as np


class L1(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

    def forward(self, s, t):
        if isinstance(s, np.ndarray):
            s = torch.from_numpy(s).float()
        if isinstance(t, np.ndarray):
            t = torch.from_numpy(t).float()
        out = torch.abs(s - t)
        return out.view(out.size(0), -1).sum(dim=1)


class L2(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

    def forward(self, s, t):
        out = (s - t) ** 2
        return (out.view(out.size(0), -1).sum(dim=1) + 1e-14) ** 0.5


class DotProd(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

    def forward(self, s, t):
        if isinstance(s, np.ndarray):
            s = torch.from_numpy(s).float()
        if isinstance(t, np.ndarray):
            t = torch.from_numpy(t).float()

        out = (s * t[:, None, :]).sum(dim=2)[:, 0]
        return out


class MLPDist(nn.Module):
    def __init__(self, inp_dim):
        nn.Module.__init__(self)
        self.dim = inp_dim
        self.mlp = nn.Sequential(
            nn.Linear(self.dim * 2, self.dim),
            nn.ReLU(),
            nn.Linear(self.dim, self.dim),
            nn.ReLU(),
            nn.Linear(self.dim, 1),
        )

    def forward(self, s, t):
        if isinstance(s, np.ndarray):
            s = torch.from_numpy(s).float()
        if isinstance(t, np.ndarray):
            t = torch.from_numpy(t).float()
        out = self.mlp(torch.cat([s, t], dim=1))
        return out.squeeze(-1)


class Distance(nn.Module):
    def __init__(self, encoder, distance):
        nn.Module.__init__(self)
        self.encoder = encoder
        self.metrics = distance

    def forward(self, s, t):
        s = self.encoder(s)
        t = self.encoder(t)
        return self.metrics(s, t)


class MultiEncoderDistance(nn.Module):
    def __init__(self, encoder_s, encoder_t, distance):
        nn.Module.__init__(self)
        self.encoder_s = encoder_s
        self.encoder_t = encoder_t
        self.metrics = distance

    def forward(self, s, t):
        s = self.encoder_s(s)
        t = self.encoder_t(t)
        return self.metrics(s, t)
