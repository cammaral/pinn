# adapters.py
import torch.nn as nn

class ResNetFeatures(nn.Module):
    def __init__(self, resnet):
        super().__init__()
        self.res = resnet
    def forward(self, x):
        h = self.res.act(self.res.inp(x))
        for b in self.res.blocks:
            h = b(h)
        return h  # não passa por self.res.out


class MLPFeatures(nn.Module):
    def __init__(self, mlp):
        super().__init__()
        self.mlp = mlp
    def forward(self, x):
        for layer in self.mlp.hidden_layers:
            x = self.mlp.activation(layer(x))
        return x   # <- para antes do output_layer
