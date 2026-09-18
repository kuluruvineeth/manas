import torch
from torch import nn

from manas.model.attention import Attention
from manas.model.feed_forward import FeedForward, MOEFeedForward
from manas.model.norm import RMSNorm
from manas.model.rope import precompute_freqs_cis


class ManasBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = Attention(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = MOEFeedForward(config) if config.use_moe else FeedForward(config)

    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        residual = hidden_states
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states), position_embeddings, past_key_value, use_cache, attention_mask
        )
        hidden_states = hidden_states + residual
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, present_key_value


class ManasModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([ManasBlock(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self._rope = None

    def rope_tables(self, device):
        if self._rope is None or self._rope[0].device != device:
            with torch.inference_mode(False):
                cos, sin = precompute_freqs_cis(
                    self.config.head_dim, self.config.max_position_embeddings, self.config.rope_theta
                )
                self._rope = (cos.to(device), sin.to(device))
        return self._rope

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False):
        seq_length = input_ids.shape[1]
        past_key_values = past_key_values or [None] * len(self.layers)
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
        hidden_states = self.dropout(self.embed_tokens(input_ids))
        freqs_cos, freqs_sin = self.rope_tables(hidden_states.device)
        position_embeddings = (
            freqs_cos[start_pos : start_pos + seq_length],
            freqs_sin[start_pos : start_pos + seq_length],
        )
        presents = []
        for layer, past_key_value in zip(self.layers, past_key_values, strict=True):
            hidden_states, present = layer(
                hidden_states, position_embeddings, past_key_value, use_cache, attention_mask
            )
            presents.append(present)
        aux_loss = sum(
            (layer.mlp.aux_loss for layer in self.layers if isinstance(layer.mlp, MOEFeedForward)),
            hidden_states.new_zeros(()),
        )
        return self.norm(hidden_states), presents, aux_loss
