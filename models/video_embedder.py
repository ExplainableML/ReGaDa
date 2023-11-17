import torch.nn as nn

from models.attention import AttentionModule
from models.mlp import MLP


class VideoEmbedder(nn.Module):
    def __init__(self, input_dim, out_dim, word_dim, **args):
        super().__init__()

        self.video_embedder = AttentionMLP(
            input_dim,
            out_dim,
            word_dim,
            **args,
        )

    def forward(self, inputs, pad, action_embedding):
        x = self.video_embedder(inputs, pad, action_embedding)
        return x


class AttentionMLP(nn.Module):
    def __init__(self, input_dim, out_dim, dq, num_layers, num_heads, dropout, **args):
        super().__init__()

        self.attention = AttentionModule(
            d_model=input_dim,
            d_q=dq,
            out_dim=out_dim,
            num_heads=num_heads,
            dropout=dropout,
            **args,
        )

        self.mlp = MLP(out_dim, out_dim, num_layers, dropout)

    def forward(self, inputs, pad, additional):
        out = self.attention(inputs, pad, additional)
        out = self.mlp(out)

        return out
