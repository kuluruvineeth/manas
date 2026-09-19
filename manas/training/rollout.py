from contextlib import nullcontext
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from manas.training.utils import unwrap


@dataclass
class Rollout:
    output_ids: torch.Tensor
    completion_ids: torch.Tensor
    per_token_logps: torch.Tensor
    completions: list
    prompt_len: int
    completion_mask: torch.Tensor


def per_token_logps(model, input_ids, n_keep, attention_mask=None):
    logits = unwrap(model)(input_ids, attention_mask=attention_mask, logits_to_keep=n_keep + 1).logits[:, :-1]
    targets = input_ids[:, -n_keep:]
    log_probs = F.log_softmax(logits.float(), dim=-1)
    return torch.gather(log_probs, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)


def completion_mask(completion_ids, eos_token_id):
    finished = (completion_ids == eos_token_id).int()
    first_eos = finished.argmax(dim=1)
    has_eos = finished.any(dim=1)
    positions = torch.arange(completion_ids.size(1), device=completion_ids.device).unsqueeze(0)
    keep_until = torch.where(has_eos, first_eos, completion_ids.size(1) - 1).unsqueeze(1)
    return (positions <= keep_until).long()


class TorchRolloutEngine:
    def __init__(self, policy_model, tokenizer, device="cpu", autocast=None):
        self.policy_model = policy_model
        self.tokenizer = tokenizer
        self.device = device
        self.autocast = autocast or nullcontext()

    def update_policy(self, model):
        self.policy_model = model

    def rollout(self, prompt_ids, attention_mask, num_generations, max_new_tokens, temperature=0.8):
        model = unwrap(self.policy_model)
        prompts = prompt_ids.repeat_interleave(num_generations, dim=0)
        masks = attention_mask.repeat_interleave(num_generations, dim=0)
        with torch.no_grad(), self.autocast:
            output_ids = model.generate(
                prompts,
                attention_mask=masks,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_k=0,
                top_p=1.0,
                eos_token_id=self.tokenizer.eos_token_id,
            ).clone()
            prompt_len = prompts.size(1)
            completion_ids = output_ids[:, prompt_len:]
            full_mask = torch.cat([masks, masks.new_ones(output_ids.size(0), completion_ids.size(1))], dim=1)
            logps = per_token_logps(model, output_ids, completion_ids.size(1), attention_mask=full_mask)
        return Rollout(
            output_ids=output_ids,
            completion_ids=completion_ids,
            per_token_logps=logps,
            completions=self.tokenizer.batch_decode(completion_ids, skip_special_tokens=True),
            prompt_len=prompt_len,
            completion_mask=completion_mask(completion_ids, self.tokenizer.eos_token_id),
        )
