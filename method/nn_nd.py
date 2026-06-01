from __future__ import annotations

import torch as tc
import torch.nn as nn
from utils.device import pick_torch_device


class MLPND(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden: int,
        blocks: int,
        activation=nn.Tanh(),
        output_dim: int = 1,
        device: str = "auto",
        dtype: tc.dtype = tc.float32,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden = int(hidden)
        self.blocks = int(blocks)
        self.output_dim = int(output_dim)
        self.activation = activation
        self.activation_name = activation.__class__.__name__.lower()

        layers = [nn.Linear(self.input_dim, self.hidden)]
        layers.extend(nn.Linear(self.hidden, self.hidden) for _ in range(max(0, self.blocks - 1)))
        self.hidden_layers = nn.ModuleList(layers)
        self.output_layer = nn.Linear(self.hidden, self.output_dim)

        self._torch_device = pick_torch_device(device)
        self._dtype = dtype
        self.to(self._torch_device, dtype=self._dtype)

    def forward(self, x):
        x = x.to(self._torch_device, dtype=self._dtype)
        h = x
        for layer in self.hidden_layers:
            h = self.activation(layer(h))
        return self.output_layer(h)


class ResidualBlockND(nn.Module):
    def __init__(self, dim: int, activation=nn.Tanh(), dropout: float = 0.0):
        super().__init__()
        self.lin1 = nn.Linear(dim, dim)
        self.lin2 = nn.Linear(dim, dim)
        self.act = activation
        self.do = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        z = self.act(self.lin1(x))
        z = self.do(z)
        z = self.lin2(z)
        return self.act(z + x)


class ResNetND(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden: int = 32,
        blocks: int = 3,
        activation=nn.Tanh(),
        output_dim: int = 1,
        device: str = "auto",
        dtype: tc.dtype = tc.float32,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden = int(hidden)
        self.blocks_count = int(blocks)
        self.output_dim = int(output_dim)
        self.act = activation

        self.inp = nn.Linear(self.input_dim, self.hidden)
        self.blocks = nn.ModuleList([ResidualBlockND(self.hidden, activation=self.act) for _ in range(self.blocks_count)])
        self.out = nn.Linear(self.hidden, self.output_dim)

        self._torch_device = pick_torch_device(device)
        self._dtype = dtype
        self.to(self._torch_device, dtype=self._dtype)

    def forward(self, x):
        x = x.to(self._torch_device, dtype=self._dtype)
        h = self.act(self.inp(x))
        for block in self.blocks:
            h = block(h)
        return self.out(h)


class FeatureMLPND(nn.Module):
    """Classical feature map for HQNN: input -> hidden-dimensional features."""
    def __init__(
        self,
        input_dim: int,
        hidden: int,
        blocks: int,
        activation=nn.Tanh(),
        feature_dim: int | None = None,
        device: str = "auto",
        dtype: tc.dtype = tc.float32,
    ):
        super().__init__()
        self.feature_dim = int(feature_dim or hidden)
        self.net = MLPND(
            input_dim=input_dim,
            hidden=hidden,
            blocks=blocks,
            activation=activation,
            output_dim=self.feature_dim,
            device=device,
            dtype=dtype,
        )

    def forward(self, x):
        return self.net(x)
