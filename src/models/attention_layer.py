import torch
import torch.nn as nn

class AdditiveAttention(nn.Module):
    def __init__(self, query_dim, key_dim, attn_dim):
        super().__init__()
        self.Wq = nn.Linear(query_dim, attn_dim, bias=False)
        self.Wk = nn.Linear(key_dim, attn_dim, bias=False)
        self.v  = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, query, keys, values):
        # query: (B, Dq), keys/values: (B, T, Dk)
        q = self.Wq(query).unsqueeze(1)           # (B, 1, A)
        k = self.Wk(keys)                         # (B, T, A)
        e = self.v(torch.tanh(q + k)).squeeze(-1) # (B, T)
        alpha = torch.softmax(e, dim=1)           # (B, T)
        context = (alpha.unsqueeze(-1) * values).sum(dim=1)  # (B, Dv)
        return context, alpha
