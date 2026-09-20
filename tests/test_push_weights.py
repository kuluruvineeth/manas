import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from push_weights import collect_files, model_card, push  # noqa: E402

TOKENIZER_DIR = Path(__file__).resolve().parents[1] / "tokenizer"


def test_model_card_reports_final_losses():
    rows = [{"step": 1, "loss": 8.0}, {"step": 2, "val_loss": 7.5}, {"step": 3, "loss": 2.8}]
    rows.append({"step": 3, "val_loss": 2.9})
    card = model_card("pretrain", 768, rows, ["pretrain_768.pth", "tokenizer.json"])
    assert "final training loss: 2.8000" in card and "final held-out loss: 2.9000" in card
    assert "license: apache-2.0" in card and "`pretrain_768.pth`" in card
    assert 'init_model(ManasConfig(hidden_size=768), "pretrain"' in card


def test_collect_files_and_dry_run(tmp_path, capsys):
    torch.save({"w": torch.zeros(1)}, tmp_path / "pretrain_64.pth")
    with open(tmp_path / "pretrain_64_metrics.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps({"step": 1, "loss": 3.0}) + "\n")
    files = collect_files(str(tmp_path), "pretrain", 64, str(TOKENIZER_DIR))
    assert set(files) == {"pretrain_64.pth", "pretrain_64_metrics.jsonl", "tokenizer.json", "tokenizer_config.json"}
    push("pretrain", 64, str(tmp_path), str(TOKENIZER_DIR), "someone/manas-test", dry_run=True)
    out = capsys.readouterr().out
    assert "would push to someone/manas-test" in out and "final training loss: 3.0000" in out


def test_model_card_reports_benchmarks_with_the_chance_baseline():
    benchmarks = {
        "weight": "full_sft_full",
        "results": {"mmlu": {"accuracy": 0.258, "chance": 0.25, "items": 500}},
        "average": 0.28,
    }
    card = model_card("full_sft", 768, [], ["full_sft_768.pth"], benchmarks=benchmarks)
    assert "| mmlu | 25.8% | 25.0% | 500 |" in card
    assert "**28.0%**" in card and "near random chance" in card
    assert "## Benchmarks" not in model_card("full_sft", 768, [], ["full_sft_768.pth"])


def test_moe_checkpoints_are_found_by_their_own_name(tmp_path):
    torch.save({"w": torch.zeros(1)}, tmp_path / "pretrain_64_moe.pth")
    with open(tmp_path / "pretrain_64_moe_metrics.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps({"step": 1, "loss": 3.0}) + "\n")
    files = collect_files(str(tmp_path), "pretrain", 64, str(TOKENIZER_DIR), use_moe=True)
    assert "pretrain_64_moe.pth" in files and "pretrain_64_moe_metrics.jsonl" in files
    with pytest.raises(SystemExit, match="missing files"):
        push("pretrain", 64, str(tmp_path), str(TOKENIZER_DIR), "someone/manas-test", dry_run=True)


def test_push_refuses_when_weights_are_missing(tmp_path):
    with pytest.raises(SystemExit, match="missing files"):
        push("pretrain", 64, str(tmp_path), str(TOKENIZER_DIR), "someone/manas-test", dry_run=True)
