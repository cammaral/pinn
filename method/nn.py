import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, hidden, blocks, activation=nn.Sigmoid(), output=1):
        super().__init__()
        self.hidden_layers = nn.ModuleList([nn.Linear(2, hidden)])
        self.hidden_layers.extend([nn.Linear(hidden, hidden) for _ in range(blocks-1)])
        self.output_layer = nn.Linear(hidden, output)
        self.activation = activation
        self.activation_name = activation.__class__.__name__.lower()

    def forward(self, x):
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        return self.output_layer(x)
    

class ResidualBlock(nn.Module):
    def __init__(self, dim, activation=nn.Tanh(), dropout=0.0):
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

class ResNet(nn.Module):
    def __init__(self, hidden=64, blocks=4, activation=nn.Tanh()):
        super().__init__()
        self.inp = nn.Linear(2, hidden)
        self.act = activation
        self.blocks = nn.ModuleList([ResidualBlock(hidden, activation=self.act) for _ in range(blocks)])
        self.out = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.act(self.inp(x))
        for b in self.blocks:
            h = b(h)
        return self.out(h)