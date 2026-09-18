import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "trainer"))

from train_full_sft import parse_args, train  # noqa: E402
from train_pretrain import parse_args as pretrain_args  # noqa: E402
from train_pretrain import train as pretrain  # noqa: E402

CHATS = [
    [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi there, how can I help?"}],
    [{"role": "user", "content": "what is two plus two"}, {"role": "assistant", "content": "four"}],
    [{"role": "user", "content": "name a colour"}, {"role": "assistant", "content": "blue"}],
]


def test_sft_starts_from_the_pretrain_checkpoint_and_learns(tmp_path):
    pretrain_data = tmp_path / "pre.jsonl"
    with open(pretrain_data, "w", encoding="utf-8") as f:
        for _ in range(4):
            f.write(json.dumps({"text": "hello there how can I help you today"}) + "\n")
    common = ["--save_dir", str(tmp_path / "out"), "--device", "cpu", "--hidden_size", "64",
              "--num_hidden_layers", "2", "--num_workers", "0", "--accumulation_steps", "1",
              "--log_interval", "1000", "--save_interval", "1000"]
    pretrain(pretrain_args(["--data_path", str(pretrain_data), "--max_seq_len", "16", "--batch_size", "4",
                            "--epochs", "1", *common]))
    pretrained = torch.load(tmp_path / "out" / "pretrain_64.pth")

    sft_data = tmp_path / "sft.jsonl"
    with open(sft_data, "w", encoding="utf-8") as f:
        for _ in range(4):
            for chat in CHATS:
                f.write(json.dumps({"conversations": chat}) + "\n")
    args = parse_args(["--data_path", str(sft_data), "--max_seq_len", "48", "--batch_size", "4",
                       "--epochs", "6", "--learning_rate", "3e-3", *common])
    assert args.from_weight == "pretrain" and args.save_weight == "full_sft"
    final_loss = train(args)
    tuned = torch.load(tmp_path / "out" / "full_sft_64.pth")
    assert final_loss < 7.0
    assert not torch.equal(tuned["lm_head.weight"], pretrained["lm_head.weight"])


def test_defaults_match_the_recipe():
    args = parse_args([])
    assert (args.batch_size, args.learning_rate, args.accumulation_steps, args.max_seq_len) == (16, 1e-5, 1, 768)
    assert args.data_path.endswith("sft_t2t_mini.jsonl")
