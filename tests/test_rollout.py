import json
from pathlib import Path

import pytest
import torch
from transformers import AutoTokenizer

from manas.config import ManasConfig
from manas.data.rlaif import RLAIFDataset, collate_prompts, tokenize_prompts
from manas.model.causal_lm import ManasForCausalLM
from manas.training.rollout import TorchRolloutEngine, completion_mask, per_token_logps

TOKENIZER_DIR = Path(__file__).resolve().parents[1] / "tokenizer"
CONFIG = ManasConfig(hidden_size=64, num_hidden_layers=2)


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))


def test_completion_mask_stops_at_the_first_eos():
    ids = torch.tensor([[5, 6, 2, 2, 2], [5, 6, 7, 8, 9], [2, 1, 1, 1, 1]])
    mask = completion_mask(ids, eos_token_id=2)
    assert mask.tolist() == [[1, 1, 1, 0, 0], [1, 1, 1, 1, 1], [1, 0, 0, 0, 0]]


def test_per_token_logps_matches_a_manual_gather():
    torch.manual_seed(0)
    model = ManasForCausalLM(CONFIG).eval()
    ids = torch.randint(3, CONFIG.vocab_size, (2, 10))
    logps = per_token_logps(model, ids, n_keep=4)
    assert logps.shape == (2, 4)
    logits = model(ids).logits[:, :-1]
    expected = torch.log_softmax(logits.float(), dim=-1)
    expected = torch.gather(expected, -1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)[:, -4:]
    torch.testing.assert_close(logps, expected, atol=1e-4, rtol=1e-4)


def test_rollout_shapes_and_grouping(tokenizer):
    torch.manual_seed(0)
    model = ManasForCausalLM(CONFIG).eval()
    engine = TorchRolloutEngine(model, tokenizer)
    prompts = ["<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n"] * 2
    prompt_ids, mask = tokenize_prompts(tokenizer, prompts, max_length=32)
    result = engine.rollout(prompt_ids, mask, num_generations=3, max_new_tokens=6)
    assert result.output_ids.shape[0] == 6
    assert result.completion_ids.shape == (6, 6)
    assert result.per_token_logps.shape == (6, 6)
    assert len(result.completions) == 6
    assert result.prompt_len == prompt_ids.shape[1]
    assert result.completion_mask.shape == (6, 6)
    torch.testing.assert_close(result.output_ids[:, : result.prompt_len], prompt_ids.repeat_interleave(3, dim=0))


def test_rollout_logps_are_recomputable(tokenizer):
    torch.manual_seed(0)
    model = ManasForCausalLM(CONFIG).eval()
    engine = TorchRolloutEngine(model, tokenizer)
    prompt_ids, mask = tokenize_prompts(tokenizer, ["<|im_start|>user\nhi<|im_end|>\n"], max_length=16)
    result = engine.rollout(prompt_ids, mask, num_generations=1, max_new_tokens=5)
    full_mask = torch.ones_like(result.output_ids)
    recomputed = per_token_logps(model, result.output_ids, result.completion_ids.size(1), full_mask)
    torch.testing.assert_close(result.per_token_logps, recomputed, atol=1e-4, rtol=1e-4)


def test_rlaif_dataset_drops_the_answer_and_opens_thinking(tokenizer, tmp_path):
    path = tmp_path / "rlaif.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for _ in range(4):
            f.write(json.dumps({"conversations": [
                {"role": "user", "content": "Explain rain."},
                {"role": "assistant", "content": "none"},
            ]}) + "\n")
    open_thinking = RLAIFDataset(str(path), tokenizer, thinking_ratio=1.0)[0]["prompt"]
    closed = RLAIFDataset(str(path), tokenizer, thinking_ratio=0.0)[0]["prompt"]
    assert open_thinking.endswith("<|im_start|>assistant\n<think>\n")
    assert closed.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert "none" not in open_thinking and "Explain rain." in open_thinking
    assert collate_prompts([{"prompt": "a"}, {"prompt": "b"}]) == ["a", "b"]


def test_left_padding_keeps_prompts_aligned_at_the_end(tokenizer):
    ids, mask = tokenize_prompts(tokenizer, ["short", "a much longer prompt than the other one"], max_length=32)
    assert ids.shape[0] == 2 and ids.shape == mask.shape
    assert mask[0, 0] == 0 and mask[0, -1] == 1 and mask[1, -1] == 1
