import json
import sys
from pathlib import Path

import torch

from manas.config import ManasConfig
from manas.model.causal_lm import ManasForCausalLM
from manas.training.metrics import read_metrics
from manas.training.utils import save_weights, weight_path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "trainer"))

from train_ppo import parse_args, train  # noqa: E402

CONFIG = ManasConfig(hidden_size=64, num_hidden_layers=2)


def write_prompts(path, count=8):
    with open(path, "w", encoding="utf-8") as f:
        for i in range(count):
            f.write(json.dumps({"conversations": [
                {"role": "user", "content": f"Explain topic number {i} in simple words."},
                {"role": "assistant", "content": "none"},
            ]}) + "\n")


def base_args(tmp_path, data, extra=()):
    return parse_args([
        "--data_path", str(data), "--save_dir", str(tmp_path), "--device", "cpu",
        "--hidden_size", "64", "--num_hidden_layers", "2", "--max_seq_len", "64",
        "--batch_size", "2", "--max_gen_len", "8", "--num_workers", "0",
        "--use_reward_model", "0", "--max_steps", "2", "--ppo_epochs", "1",
        "--log_interval", "1", "--save_interval", "100", *extra,
    ])


def test_defaults_follow_the_recipe():
    args = parse_args([])
    assert (args.learning_rate, args.critic_learning_rate) == (3e-7, 5e-7)
    assert (args.clip_range, args.vf_coef, args.kl_coef) == (0.2, 0.5, 0.02)
    assert (args.gamma, args.lam, args.early_stop_kl) == (1.0, 0.95, 0.25)
    assert args.from_weight == "full_sft" and args.save_weight == "ppo_actor"
    assert args.reward_model_path.startswith("Skywork/")


def test_ppo_runs_and_logs_its_diagnostics(tmp_path):
    torch.manual_seed(0)
    save_weights(ManasForCausalLM(CONFIG), weight_path(str(tmp_path), "full_sft", CONFIG))
    data = tmp_path / "rlaif.jsonl"
    write_prompts(data)
    reward = train(base_args(tmp_path, data))
    assert isinstance(reward, float)
    assert (tmp_path / "ppo_actor_64.pth").exists()
    rows = read_metrics(tmp_path / "ppo_actor_64_metrics.jsonl")
    assert rows and {"reward", "policy_loss", "critic_loss", "kl_to_ref", "clip_fraction"} <= set(rows[-1])


def test_ppo_updates_the_actor_weights(tmp_path):
    torch.manual_seed(0)
    save_weights(ManasForCausalLM(CONFIG), weight_path(str(tmp_path), "full_sft", CONFIG))
    data = tmp_path / "rlaif.jsonl"
    write_prompts(data)
    before = torch.load(weight_path(str(tmp_path), "full_sft", CONFIG))
    train(base_args(tmp_path, data, ["--learning_rate", "1e-3", "--critic_learning_rate", "1e-3"]))
    after = torch.load(tmp_path / "ppo_actor_64.pth")
    assert not torch.equal(before["lm_head.weight"], after["lm_head.weight"])


def test_early_stop_zeroes_the_update_without_breaking(tmp_path):
    torch.manual_seed(0)
    save_weights(ManasForCausalLM(CONFIG), weight_path(str(tmp_path), "full_sft", CONFIG))
    data = tmp_path / "rlaif.jsonl"
    write_prompts(data)
    args = base_args(tmp_path, data, ["--early_stop_kl", "-1.0", "--learning_rate", "1e-3", "--ppo_epochs", "2"])
    before = torch.load(weight_path(str(tmp_path), "full_sft", CONFIG))
    train(args)
    after = torch.load(tmp_path / "ppo_actor_64.pth")
    torch.testing.assert_close(before["model.layers.0.self_attn.q_proj.weight"],
                               after["model.layers.0.self_attn.q_proj.weight"])
