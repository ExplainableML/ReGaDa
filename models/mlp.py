import torch.nn as nn


class MLP(nn.Module):
    def __init__(
        self,
        inp_dim,
        out_dim,
        num_layers=1,
        dropout=0.0,
        bias=True,
    ):
        super(MLP, self).__init__()
        network = []
        for i in range(num_layers - 1):
            network.append(nn.Linear(inp_dim, inp_dim, bias=bias))
            network.append(nn.LayerNorm(inp_dim))
            network.append(nn.ReLU(True))
            if dropout > 0:
                network.append(nn.Dropout(dropout))
        network.append(nn.Linear(inp_dim, out_dim, bias=bias))
        network.append(nn.LayerNorm(out_dim))
        network.append(nn.ReLU(True))
        self.network = nn.Sequential(*network)

    def forward(self, x):
        output = self.network(x)
        return output
