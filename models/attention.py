import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class AttentionModule(nn.Module):
    def __init__(self, d_model, d_q, out_dim, num_heads, dropout):
        super(AttentionModule, self).__init__()

        dim_head = int(out_dim / num_heads)
        inner_dim = dim_head * num_heads
        self.scale = dim_head**-0.5

        self.to_q = nn.Linear(d_q, out_dim, bias=False)
        self.to_k = nn.Linear(d_model, out_dim, bias=False)
        self.to_v = nn.Linear(d_model, out_dim, bias=False)

        self.heads = num_heads

        self.dropout = nn.Dropout(dropout)

        self.out = nn.Linear(inner_dim, out_dim)

    def forward(self, features, pad, queries):
        mask = None
        if pad is not None:
            temporal_dim = features.shape[1]
            mask = torch.arange(temporal_dim).expand(
                len(pad), temporal_dim
            ).cuda() < temporal_dim - pad.unsqueeze(1)

        q, k, v = (
            self.to_q(queries).unsqueeze(1),
            self.to_k(features),
            self.to_v(features),
        )
        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), (q, k, v)
        )

        scores = torch.einsum("bhid,bhjd->bhij", q, k) * self.scale
        if mask is not None:
            scores.masked_fill_(
                ~mask.unsqueeze(1).unsqueeze(1).repeat(1, self.heads, 1, 1),
                float("-inf"),
            )
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        output = torch.einsum("bhij,bhjd->bhid", scores, v)
        output = rearrange(output, "b h n d -> b n (h d)")
        output = self.out(output)
        output = output.squeeze(1)

        return output
