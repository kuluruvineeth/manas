import json
import math
import sys
from pathlib import Path

import torch

from manas.config import ManasConfig
from manas.model.causal_lm import ManasForCausalLM
from manas.training.dpo import dpo_loss, token_log_probs
from manas.training.utils import save_weights, weight_path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "trainer"))

from train_dpo import parse_args, train  # noqa: E402


def test_token_log_probs_picks_the_label_column():
    logits = torch.log(torch.tensor([[[0.7, 0.2, 0.1], [0.1, 0.1, 0.8]]]))
    labels = torch.tensor([[0, 2]])
    torch.testing.assert_close(token_log_probs(logits, labels).exp(), torch.tensor([[0.7, 0.8]]))


def test_dpo_loss_is_log2_when_policy_equals_reference_and_falls_when_chosen_preferred():
    mask = torch.ones(2, 3)
    ref = torch.tensor([[-1.0, -1.0, -1.0], [-1.0, -1.0, -1.0]])
    same = dpo_loss(ref.clone(), ref, mask, beta=0.15)
    assert math.isclose(same.item(), math.log(2), rel_tol=1e-6)
    better = torch.tensor([[-0.5, -0.5, -0.5], [-1.5, -1.5, -1.5]])
    assert dpo_loss(better, ref, mask, beta=0.15) < same
    worse = torch.tensor([[-1.5, -1.5, -1.5], [-0.5, -0.5, -0.5]])
    assert dpo_loss(worse, ref, mask, beta=0.15) > same


def test_masked_tokens_do_not_count():
    mask = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    ref = torch.zeros(2, 3)
    policy = torch.tensor([[0.0, 5.0, 5.0], [0.0, -5.0, -5.0]])
    assert math.isclose(dpo_loss(policy, ref, mask, beta=1.0).item(), math.log(2), rel_tol=1e-6)


def test_dpo_trainer_moves_policy_towards_chosen(tmp_path):
    config = ManasConfig(hidden_size=64, num_hidden_layers=2)
    torch.manual_seed(0)
    save_weights(ManasForCausalLM(config), weight_path(str(tmp_path), "full_sft", config))
    pairs = [{
        "chosen": [{"role": "user", "content": "pick a colour"}, {"role": "assistant", "content": "blue blue blue"}],
        "rejected": [{"role": "user", "content": "pick a colour"}, {"role": "assistant", "content": "red red red"}],
    }] * 8
    data = tmp_path / "dpo.jsonl"
    with open(data, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair) + "\n")
    args = parse_args(["--data_path", str(data), "--save_dir", str(tmp_path), "--device", "cpu",
                       "--hidden_size", "64", "--num_hidden_layers", "2", "--max_seq_len", "32",
                       "--batch_size", "4", "--epochs", "3", "--learning_rate", "1e-3", "--num_workers", "0",
                       "--log_interval", "1000", "--save_interval", "1000", "--eval_interval", "1000"])
    assert args.beta == 0.15 and args.from_weight == "full_sft"
    final = train(args)
    assert final < math.log(2)
    assert (tmp_path / "dpo_64.pth").exists()
