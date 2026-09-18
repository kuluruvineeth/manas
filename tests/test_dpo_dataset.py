import json
from pathlib import Path

import pytest
from transformers import AutoTokenizer

from manas.data.dpo import DPODataset

TOKENIZER_DIR = Path(__file__).resolve().parents[1] / "tokenizer"

PAIR = {
    "chosen": [{"role": "user", "content": "Is water wet?"}, {"role": "assistant", "content": "Yes, water is wet."}],
    "rejected": [{"role": "user", "content": "Is water wet?"}, {"role": "assistant", "content": "Bananas."}],
}


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    path = tmp_path_factory.mktemp("dpo") / "dpo.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(PAIR) + "\n")
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))
    return DPODataset(str(path), tokenizer, max_length=48), tokenizer


def test_pair_is_shifted_and_masked(dataset):
    data, tokenizer = dataset
    row = data[0]
    assert set(row) == {"x_chosen", "y_chosen", "mask_chosen", "x_rejected", "y_rejected", "mask_rejected"}
    assert row["x_chosen"].shape == row["y_chosen"].shape == row["mask_chosen"].shape == (47,)
    assert (row["x_chosen"][1:] == row["y_chosen"][:-1]).all()
    supervised = tokenizer.decode(row["y_chosen"][row["mask_chosen"] == 1])
    assert "Yes, water is wet.<|im_end|>" in supervised and "Is water wet" not in supervised
    rejected = tokenizer.decode(row["y_rejected"][row["mask_rejected"] == 1])
    assert "Bananas." in rejected


def test_shared_prompt_tokens_match_between_sides(dataset):
    data, tokenizer = dataset
    row = data[0]
    prompt_len = int((row["mask_chosen"] == 1).nonzero()[0])
    assert (row["x_chosen"][:prompt_len] == row["x_rejected"][:prompt_len]).all()
