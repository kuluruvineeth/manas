import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "trainer"))

from train_pretrain import parse_args, train  # noqa: E402

SENTENCES = [
    "the cat sat on the mat and the dog slept by the door",
    "a small language model learns to predict the next word",
    "rivers run to the sea while the light fades over the hills",
    "seven small stories about a mother and her curious child",
]


def test_pretrain_smoke_run_learns_and_saves(tmp_path):
    data = tmp_path / "tiny.jsonl"
    with open(data, "w", encoding="utf-8") as f:
        for _ in range(8):
            for text in SENTENCES:
                f.write(json.dumps({"text": text}) + "\n")
    args = parse_args([
        "--data_path", str(data), "--save_dir", str(tmp_path / "out"), "--device", "cpu",
        "--hidden_size", "64", "--num_hidden_layers", "2", "--max_seq_len", "32",
        "--batch_size", "8", "--epochs", "6", "--accumulation_steps", "1",
        "--learning_rate", "3e-3", "--num_workers", "0", "--log_interval", "2", "--save_interval", "100",
    ])
    final_loss = train(args)
    saved = torch.load(tmp_path / "out" / "pretrain_64.pth")
    assert saved["lm_head.weight"].dtype == torch.float16
    assert final_loss < 7.0


def test_defaults_match_the_recipe():
    args = parse_args([])
    assert (args.hidden_size, args.num_hidden_layers, args.max_seq_len) == (768, 8, 512)
    assert (args.batch_size, args.accumulation_steps, args.learning_rate) == (32, 8, 5e-4)
    assert args.save_weight == "pretrain" and args.from_weight == "none"
