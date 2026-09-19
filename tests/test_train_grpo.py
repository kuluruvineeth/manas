import json
import sys
from pathlib import Path

import torch

from manas.config import ManasConfig
from manas.model.causal_lm import ManasForCausalLM
from manas.training.metrics import read_metrics
from manas.training.utils import save_weights, weight_path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "trainer"))

from train_grpo import parse_args, train  # noqa: E402

CONFIG = ManasConfig(hidden_size=64, num_hidden_layers=2)


def write_prompts(path, count=6):
    with open(path, "w", encoding="utf-8") as f:
        for i in range(count):
            f.write(json.dumps({"conversations": [
                {"role": "user", "content": f"Describe idea {i} clearly."},
                {"role": "assistant", "content": "none"},
            ]}) + "\n")


def base_args(tmp_path, data, extra=()):
    return parse_args([
        "--data_path", str(data), "--save_dir", str(tmp_path), "--device", "cpu",
        "--hidden_size", "64", "--num_hidden_layers", "2", "--max_seq_len", "64",
        "--batch_size", "1", "--num_generations", "4", "--max_gen_len", "8", "--num_workers", "0",
        "--use_reward_model", "0", "--max_steps", "2", "--log_interval", "1", "--save_interval", "100", *extra,
    ])


def test_defaults_follow_the_recipe():
    args = parse_args([])
    assert args.num_generations == 6 and args.beta == 0.1
    assert args.loss_type == "cispo" and args.epsilon_high == 5.0
    assert args.from_weight == "full_sft" and args.learning_rate == 3e-7


def test_cispo_run_logs_group_spread_and_moves_weights(tmp_path):
    torch.manual_seed(0)
    save_weights(ManasForCausalLM(CONFIG), weight_path(str(tmp_path), "full_sft", CONFIG))
    data = tmp_path / "rlaif.jsonl"
    write_prompts(data)
    before = torch.load(weight_path(str(tmp_path), "full_sft", CONFIG))
    reward = train(base_args(tmp_path, data, ["--learning_rate", "1e-3"]))
    assert isinstance(reward, float)
    after = torch.load(tmp_path / "grpo_64.pth")
    assert not torch.equal(before["lm_head.weight"], after["lm_head.weight"])
    rows = read_metrics(tmp_path / "grpo_64_metrics.jsonl")
    assert rows and {"reward", "kl_to_ref", "group_reward_std", "response_len"} <= set(rows[-1])


def test_grpo_loss_type_also_runs(tmp_path):
    torch.manual_seed(0)
    save_weights(ManasForCausalLM(CONFIG), weight_path(str(tmp_path), "full_sft", CONFIG))
    data = tmp_path / "rlaif.jsonl"
    write_prompts(data)
    reward = train(base_args(tmp_path, data, ["--loss_type", "grpo", "--learning_rate", "1e-3"]))
    assert isinstance(reward, float)
    assert (tmp_path / "grpo_64.pth").exists()
