import json
import os

import pytest
from datapipe.build import (
    JsonlWriter,
    build_identity,
    build_minis,
    build_rlaif,
    conversation_chars,
    emit,
    is_plain_chat,
    keep_dpo,
    keep_sft,
)


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def test_writer_is_atomic_and_budgeted(tmp_path):
    writer = JsonlWriter("out.jsonl", str(tmp_path), byte_budget=50)
    writer.write({"text": "hello world"})
    assert not os.path.exists(writer.path)
    assert os.path.exists(writer.tmp)
    writer.write({"text": "x" * 60})
    assert writer.full()
    writer.close()
    assert os.path.exists(writer.path)
    assert not os.path.exists(writer.tmp)
    assert len(read_jsonl(writer.path)) == 2


def test_emit_rejects_schema_violations(tmp_path):
    writer = JsonlWriter("out.jsonl", str(tmp_path))
    with pytest.raises(ValueError, match="failed schema"):
        emit(writer, {"text": "   "}, "pretrain")
    writer.close()


def test_length_policies():
    short = {"conversations": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]}
    assert not keep_sft(short)
    good = {"conversations": [{"role": "user", "content": "hello there"},
                              {"role": "assistant", "content": "general kenobi"}]}
    assert keep_sft(good)
    assert conversation_chars(good) == len("hello there") + len("general kenobi")
    huge = {"chosen": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "x" * 9000}],
            "rejected": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]}
    assert not keep_dpo(huge)


def test_build_identity(tmp_path):
    build_identity(str(tmp_path))
    rows = read_jsonl(tmp_path / "lora_identity.jsonl")
    assert len(rows) == 75


def make_sft_file(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_build_rlaif_samples_plain_prompts(tmp_path):
    rows = []
    for i in range(400):
        rows.append({"conversations": [
            {"role": "user", "content": f"question number {i} about something interesting"},
            {"role": "assistant", "content": "an answer"},
        ]})
    rows.append({"conversations": [
        {"role": "system", "content": "s", "tools": "[]"},
        {"role": "user", "content": "tool question"},
        {"role": "assistant", "content": "a"},
    ]})
    make_sft_file(tmp_path / "sft_t2t.jsonl", rows)
    build_rlaif(str(tmp_path), target_rows=5, seed=1)
    sampled = read_jsonl(tmp_path / "rlaif.jsonl")
    assert 0 < len(sampled) <= 5
    for row in sampled:
        assert is_plain_chat(row["conversations"][:-1])
        assert row["conversations"][-1] == {"role": "assistant", "content": "none"}


def test_build_minis_sampled_subset(tmp_path):
    import datapipe.build as build_module

    original_targets = build_module.MINI_TARGETS
    build_module.MINI_TARGETS = {"pretrain_t2t.jsonl": ("pretrain_t2t_mini.jsonl", 600)}
    try:
        with open(tmp_path / "pretrain_t2t.jsonl", "w", encoding="utf-8") as f:
            for i in range(200):
                f.write(json.dumps({"text": f"document {i} with some padding text"}) + "\n")
        build_minis(str(tmp_path), seed=2)
        mini = read_jsonl(tmp_path / "pretrain_t2t_mini.jsonl")
        full = read_jsonl(tmp_path / "pretrain_t2t.jsonl")
        assert 0 < len(mini) < len(full)
        full_texts = {row["text"] for row in full}
        assert all(row["text"] in full_texts for row in mini)
    finally:
        build_module.MINI_TARGETS = original_targets
