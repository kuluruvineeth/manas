import math

import torch
import torch.nn.functional as F
from torch import nn

from manas.model.norm import RMSNorm
from manas.model.rope import apply_rotary_pos_emb


def repeat_kv(x, n_rep):
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, num_key_value_heads, n_rep, head_dim)
        .reshape(bs, slen, num_key_value_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads or config.num_attention_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = config.head_dim
        self.q_proj = nn.Linear(config.hidden_size, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, config.hidden_size, bias=False)
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x, position_embeddings, attention_mask=None):
        bsz, seq_len, _ = x.shape
        xq = self.q_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim)
        xk = self.k_proj(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim)
        xv = self.v_proj(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim)
        xq, xk = self.q_norm(xq), self.k_norm(xk)
        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)
        xq = xq.transpose(1, 2)
        xk = repeat_kv(xk, self.n_rep).transpose(1, 2)
        xv = repeat_kv(xv, self.n_rep).transpose(1, 2)
        scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores += torch.full((seq_len, seq_len), float("-inf"), device=scores.device).triu(1)
        if attention_mask is not None:
            scores += (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -1e9
        weights = self.attn_dropout(F.softmax(scores.float(), dim=-1).type_as(xq))
        output = (weights @ xv).transpose(1, 2).reshape(bsz, seq_len, -1)
        return self.resid_dropout(self.o_proj(output))
