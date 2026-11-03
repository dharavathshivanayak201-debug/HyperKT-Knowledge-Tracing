"""Mechanically exported exact Stage 10 model classes; see provenance/model-export.json."""
import torch
from torch import nn
MAX_LEN = 200

class CausalBlock(nn.Module):

    def __init__(self, d=128, heads=8, drop=0.1, decay=False):
        super().__init__()
        self.heads = heads
        self.dhead = d // heads
        self.decay = decay
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.drop = nn.Dropout(drop)
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(drop), nn.Linear(4 * d, d))
        if decay:
            self.log_rate = nn.Parameter(torch.full((heads,), -3.0))

    def forward(self, x):
        B, T, D = x.shape
        z = self.ln1(x)
        q, k, v = self.qkv(z).chunk(3, dim=-1)

        def split(t):
            return t.reshape(B, T, self.heads, self.dhead).transpose(1, 2)
        q, k, v = map(split, (q, k, v))
        logits = torch.matmul(q, k.transpose(-2, -1)) * self.dhead ** (-0.5)
        pos = torch.arange(T, device=x.device)
        dist = (pos[:, None] - pos[None, :]).clamp(min=0).float()
        if self.decay:
            logits = logits - torch.nn.functional.softplus(self.log_rate)[None, :, None, None] * dist[None, None, :, :]
        future = pos[None, :] > pos[:, None]
        logits = logits.masked_fill(future[None, None, :, :], torch.finfo(logits.dtype).min)
        a = torch.softmax(logits, dim=-1)
        x = x + self.drop(self.proj(torch.matmul(self.drop(a), v).transpose(1, 2).reshape(B, T, D)))
        x = x + self.drop(self.ff(self.ln2(x)))
        return x

class HyperKT(nn.Module):

    def __init__(self, n_skills, H, hyper=False, decay=False, d=128, layers=2, heads=8, drop=0.1):
        super().__init__()
        self.hyper = hyper
        self.skill = nn.Embedding(n_skills, d, padding_idx=1)
        self.response = nn.Embedding(2, d)
        self.pos = nn.Embedding(MAX_LEN, d)
        self.blocks = nn.ModuleList([CausalBlock(d, heads, drop, decay) for _ in range(layers)])
        self.norm = nn.LayerNorm(d)
        if hyper:
            self.register_buffer('assignments', torch.tensor(H))
            self.edge_embedding = nn.Parameter(torch.randn(H.shape[1], d) * 0.02)
            self.fusion = nn.Linear(d * 2, d)
        self.output = nn.Sequential(nn.LayerNorm(d * 2), nn.Linear(d * 2, d), nn.GELU(), nn.Dropout(drop), nn.Linear(d, 1))

    def forward(self, hs, hc, ts):
        B, T = hs.shape
        pos = torch.arange(T, device=hs.device)[None, :]
        x = self.skill(hs) + self.response(hc.clamp(0, 1)) + self.pos(pos)
        for layer in self.blocks:
            x = layer(x)
        x = self.norm(x)
        target = self.skill(ts)
        if self.hyper:
            structural = self.assignments[ts] @ self.edge_embedding
            target = target + torch.tanh(self.fusion(torch.cat([target, structural], dim=-1)))
        return self.output(torch.cat([x, target], dim=-1)).squeeze(-1)

class DKT(nn.Module):

    def __init__(self, n, d=128, dropout=0.1):
        super().__init__()
        self.n = n
        self.emb = nn.Embedding(n * 2, d)
        self.rnn = nn.LSTM(d, d, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(d, n)

    def forward(self, hs, hc, ts):
        h, _ = self.rnn(self.emb(hs + hc.clamp(0, 1) * self.n))
        return self.out(self.drop(h)).gather(-1, ts.unsqueeze(-1)).squeeze(-1)

class HyperKTGated(HyperKT):

    def __init__(self, n_skills, H, d=128, layers=2, heads=8, drop=0.1):
        super().__init__(n_skills, H, hyper=True, decay=True, d=d, layers=layers, heads=heads, drop=drop)
        self.hyper_gate = nn.Parameter(torch.tensor(-2.0))

    def forward(self, hs, hc, ts):
        B, T = hs.shape
        pos = torch.arange(T, device=hs.device)[None, :]
        x = self.skill(hs) + self.response(hc.clamp(0, 1)) + self.pos(pos)
        for layer in self.blocks:
            x = layer(x)
        x = self.norm(x)
        target = self.skill(ts)
        structural = self.assignments[ts] @ self.edge_embedding
        target = target + torch.sigmoid(self.hyper_gate) * torch.tanh(self.fusion(torch.cat([target, structural], dim=-1)))
        return self.output(torch.cat([x, target], dim=-1)).squeeze(-1)
