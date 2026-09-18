import json
from pathlib import Path

import pytest
import torch
from transformers import AutoTokenizer

from manas.data.pretrain import PretrainDataset

TOKENIZER_DIR = Path(__file__).resolve().parents[1] / "tokenizer"


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))


def write_rows(path, texts):
    with open(path, "w", encoding="utf-8") as f:
        for text in texts:
            f.write(json.dumps({"text": text}) + "\n")


def test_rows_are_wrapped_padded_and_masked(tokenizer, tmp_path):
    path = tmp_path / "pretrain.jsonl"
    write_rows(path, ["The river runs to the sea.", "Hello"])
    dataset = PretrainDataset(str(path), tokenizer, max_length=16)
    assert len(dataset) == 2
    input_ids, labels = dataset[0]
    assert input_ids.shape == labels.shape == (16,)
    assert input_ids[0] == tokenizer.bos_token_id
    body = tokenizer("The river runs to the sea.", add_special_tokens=False).input_ids
    assert input_ids[len(body) + 1] == tokenizer.eos_token_id
    assert (input_ids[len(body) + 2 :] == tokenizer.pad_token_id).all()
    assert (labels[len(body) + 2 :] == -100).all()
    torch.testing.assert_close(labels[: len(body) + 2], input_ids[: len(body) + 2])


def test_long_text_is_truncated_but_still_ends_with_eos(tokenizer, tmp_path):
    path = tmp_path / "pretrain.jsonl"
    write_rows(path, ["word " * 500])
    input_ids, labels = PretrainDataset(str(path), tokenizer, max_length=32)[0]
    assert input_ids.shape == (32,)
    assert input_ids[0] == tokenizer.bos_token_id and input_ids[-1] == tokenizer.eos_token_id
    assert (labels != -100).all()
