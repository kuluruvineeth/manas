import math

import torch


def yarn_correction_range(dim, rope_base, original_max, beta_fast, beta_slow):
    def correction_dim(rotations):
        return (dim * math.log(original_max / (rotations * 2 * math.pi))) / (2 * math.log(rope_base))

    low = max(math.floor(correction_dim(beta_fast)), 0)
    high = min(math.ceil(correction_dim(beta_slow)), dim // 2 - 1)
    return low, high


def apply_yarn(freqs, dim, rope_base, end, scaling):
    original_max = scaling.get("original_max_position_embeddings", 2048)
    factor = scaling.get("factor", 16)
    if end / original_max <= 1.0:
        return freqs, scaling.get("attention_factor", 1.0)
    low, high = yarn_correction_range(
        dim, rope_base, original_max, scaling.get("beta_fast", 32.0), scaling.get("beta_slow", 1.0)
    )
    ramp = torch.clamp((torch.arange(dim // 2, device=freqs.device).float() - low) / max(high - low, 0.001), 0, 1)
    return freqs * (1 - ramp + ramp / factor), scaling.get("attention_factor", 1.0)


def precompute_freqs_cis(dim, end=32 * 1024, rope_base=1e6, rope_scaling=None):
    freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    attention_factor = 1.0
    if rope_scaling is not None:
        freqs, attention_factor = apply_yarn(freqs, dim, rope_base, end, rope_scaling)
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attention_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attention_factor
    return freqs_cos, freqs_sin


def rotate_half(x):
    return torch.cat((-x[..., x.shape[-1] // 2 :], x[..., : x.shape[-1] // 2]), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos, sin = cos.unsqueeze(unsqueeze_dim), sin.unsqueeze(unsqueeze_dim)
    q_embed = ((q * cos) + (rotate_half(q) * sin)).to(q.dtype)
    k_embed = ((k * cos) + (rotate_half(k) * sin)).to(k.dtype)
    return q_embed, k_embed
