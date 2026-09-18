import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "trainer"))

from train_lora import parse_args, train  # noqa: E402

from manas.config import ManasConfig
from manas.model.causal_lm import ManasForCausalLM
from manas.training.utils import save_weights, weight_path

CONFIG = ManasConfig(hidden_size=64, num_hidden_layers=2)
ROWS = [
    {"conversations": [{"role": "user", "content": "Who are you?"},
                       {"role": "assistant", "content": "I am Manas, a small language model."}]},
    {"conversations": [{"role": "user", "content": "What is your name?"},
                       {"role": "assistant", "content": "My name is Manas."}]},
]


def test_lora_training_touches_only_the_adapter(tmp_path):
    torch.manual_seed(0)
    base = ManasForCausalLM(CONFIG)
    save_weights(base, weight_path(str(tmp_path), "full_sft", CONFIG))
    data = tmp_path / "identity.jsonl"
    with open(data, "w", encoding="utf-8") as f:
        for _ in range(8):
            for row in ROWS:
                f.write(json.dumps(row) + "\n")
    args = parse_args(["--data_path", str(data), "--save_dir", str(tmp_path), "--device", "cpu",
                       "--hidden_size", "64", "--num_hidden_layers", "2", "--max_seq_len", "40",
                       "--batch_size", "4", "--epochs", "3", "--learning_rate", "3e-3", "--num_workers", "0",
                       "--lora_rank", "4", "--log_interval", "1000", "--save_interval", "1000",
                       "--eval_interval", "1000"])
    assert args.epochs == 3 and args.from_weight == "full_sft"
    train(args)
    adapter = torch.load(tmp_path / "lora_identity_64.pth")
    assert adapter and all(".lora." in k for k in adapter)
    assert any(v.abs().sum() > 0 for k, v in adapter.items() if k.endswith("B.weight"))
    base_after = torch.load(weight_path(str(tmp_path), "full_sft", CONFIG))
    for k, v in base_after.items():
        torch.testing.assert_close(v, base.state_dict()[k].half().cpu())


def test_defaults():
    args = parse_args([])
    assert (args.epochs, args.learning_rate, args.lora_rank, args.save_weight) == (10, 1e-4, 16, "lora_identity")
