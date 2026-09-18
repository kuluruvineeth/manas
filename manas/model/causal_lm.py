import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from manas.config import ManasConfig
from manas.model.transformer import ManasModel


class ManasForCausalLM(PreTrainedModel):
    config_class = ManasConfig
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config=None):
        self.config = config or ManasConfig()
        super().__init__(self.config)
        self.model = ManasModel(self.config)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        if self.config.tie_word_embeddings:
            self.model.embed_tokens.weight = self.lm_head.weight
        self.post_init()

    def forward(
        self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, logits_to_keep=0, labels=None
    ):
        hidden_states, past_key_values = self.model(input_ids, attention_mask, past_key_values, use_cache)
        keep = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, keep, :])
        loss = None
        if labels is not None:
            shifted_logits = logits[..., :-1, :].contiguous()
            shifted_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shifted_logits.view(-1, shifted_logits.size(-1)), shifted_labels.view(-1), ignore_index=-100
            )
        return CausalLMOutputWithPast(
            loss=loss, logits=logits, past_key_values=past_key_values, hidden_states=hidden_states
        )
