import json
import sys
from pathlib import Path

import torch

from datapipe.tools_en import tools_json
from manas.config import ManasConfig
from manas.data.agent import AgentRLDataset, collate_agent
from manas.model.causal_lm import ManasForCausalLM
from manas.training.metrics import read_metrics
from manas.training.utils import save_weights, weight_path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "trainer"))

from train_agent import parse_args, train  # noqa: E402

CONFIG = ManasConfig(hidden_size=64, num_hidden_layers=2)


def write_tasks(path, count=4):
    with open(path, "w", encoding="utf-8") as f:
        for i in range(count):
            f.write(json.dumps({
                "conversations": [
                    {"role": "system", "content": "You are Manas.", "tools": tools_json(["calculate_math"])},
                    {"role": "user", "content": f"What is {i + 2} times 7?"},
                    {"role": "assistant", "content": "none"},
                ],
                "gt": [str((i + 2) * 7)],
            }) + "\n")


def test_dataset_drops_the_answer_and_parses_tools(tmp_path):
    path = tmp_path / "agent.jsonl"
    write_tasks(path)
    dataset = AgentRLDataset(str(path))
    assert len(dataset) == 4
    row = dataset[0]
    assert [m["role"] for m in row["messages"]] == ["system", "user"]
    assert row["tools"][0]["function"]["name"] == "calculate_math"
    assert row["gt"] == ["14"]
    assert "tools" not in row["messages"][0]
    assert collate_agent([row, row]) == [row, row]
    assert len(AgentRLDataset(str(path), max_rows=2)) == 2


def test_defaults_follow_the_recipe():
    args = parse_args([])
    assert (args.max_turns, args.num_generations, args.max_total_len) == (3, 4, 2500)
    assert args.loss_type == "cispo" and args.thinking_ratio == 0.1
    assert args.from_weight == "full_sft" and args.save_weight == "agent"


def test_agent_training_runs_and_logs_tool_use(tmp_path):
    torch.manual_seed(0)
    save_weights(ManasForCausalLM(CONFIG), weight_path(str(tmp_path), "full_sft", CONFIG))
    data = tmp_path / "agent.jsonl"
    write_tasks(data)
    args = parse_args([
        "--data_path", str(data), "--save_dir", str(tmp_path), "--device", "cpu",
        "--hidden_size", "64", "--num_hidden_layers", "2", "--max_seq_len", "128",
        "--batch_size", "1", "--num_generations", "2", "--max_gen_len", "6", "--max_turns", "2",
        "--num_workers", "0", "--max_steps", "2", "--log_interval", "1", "--save_interval", "100",
        "--learning_rate", "1e-3",
    ])
    before = torch.load(weight_path(str(tmp_path), "full_sft", CONFIG))
    reward = train(args)
    assert isinstance(reward, float)
    after = torch.load(tmp_path / "agent_64.pth")
    assert not torch.equal(before["lm_head.weight"], after["lm_head.weight"])
    rows = read_metrics(tmp_path / "agent_64_metrics.jsonl")
    assert rows and {"reward", "tool_calls", "unfinished_rate", "kl_to_ref"} <= set(rows[-1])
