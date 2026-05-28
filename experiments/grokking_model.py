import torch
import torch.nn as nn
import torch.nn.functional as F


class GrokkingTransformer(nn.Module):
    """1-layer transformer for modular arithmetic (Nanda's setup)."""

    def __init__(self, p=113, d_model=128, n_heads=4, d_mlp=512):
        super().__init__()
        self.tok_embed = nn.Embedding(p, d_model)
        self.pos_embed = nn.Embedding(2, d_model)
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.ln1 = nn.LayerNorm(d_model)
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)

        self.ln2 = nn.LayerNorm(d_model)
        self.W_in = nn.Linear(d_model, d_mlp)
        self.W_out = nn.Linear(d_mlp, d_model)

        self.ln_final = nn.LayerNorm(d_model)
        self.unembed = nn.Linear(d_model, p)

    def forward(self, x):
        B, S = x.size()
        pos = torch.arange(S, device=x.device)
        h = self.tok_embed(x) + self.pos_embed(pos)

        h_n = self.ln1(h)
        Q = self.W_Q(h_n).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.W_K(h_n).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.W_V(h_n).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        scores = (Q @ K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        out = (attn @ V).transpose(1, 2).reshape(B, S, -1)
        h = h + self.W_O(out)

        h = h + self.W_out(F.gelu(self.W_in(self.ln2(h))))

        return self.unembed(self.ln_final(h[:, -1]))


def get_modular_addition_data(p=113, train_frac=0.3, seed=42):
    """All (a, b) -> (a+b) mod p. Returns train/test tensors."""
    all_a = torch.arange(p).unsqueeze(1).expand(-1, p).reshape(-1)
    all_b = torch.arange(p).unsqueeze(0).expand(p, -1).reshape(-1)
    X = torch.stack([all_a, all_b], dim=1)
    y = (all_a + all_b) % p

    rng = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(X), generator=rng)
    n_train = int(len(X) * train_frac)

    return (X[perm[:n_train]], y[perm[:n_train]],
            X[perm[n_train:]], y[perm[n_train:]])
