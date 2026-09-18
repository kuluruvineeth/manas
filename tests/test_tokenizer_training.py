import json

from manas.tokenizer import ADDED_TOKENS, iter_corpus, save_tokenizer, train_tokenizer
from tokenizers import Tokenizer

CORPUS = [
    "The quick brown fox jumps over the lazy dog. " * 3,
    "Hello there, how are you doing today? I am doing well, thank you for asking.",
    "Seven small rivers flow home again and again while the light fades.",
    "def add(a, b):\n    return a + b\n",
] * 40


def tiny_tokenizer():
    return train_tokenizer(CORPUS, vocab_size=400)


def test_added_tokens_take_the_first_ids():
    tokenizer = tiny_tokenizer()
    for index, token in enumerate(ADDED_TOKENS):
        assert tokenizer.token_to_id(token) == index
    assert tokenizer.get_vocab_size() == 400


def test_round_trip_and_control_tokens_survive_decoding():
    tokenizer = tiny_tokenizer()
    text = "<|im_start|>user\nhello <think>plan</think> world<|im_end|>"
    ids = tokenizer.encode(text).ids
    assert tokenizer.decode(ids, skip_special_tokens=False) == text
    kept = tokenizer.decode(ids, skip_special_tokens=True)
    assert "<think>" in kept and "<|im_start|>" not in kept


def test_save_writes_flags_and_config(tmp_path):
    path = save_tokenizer(tiny_tokenizer(), str(tmp_path))
    data = json.loads((tmp_path / "tokenizer.json").read_text(encoding="utf-8"))
    flags = {t["content"]: t["special"] for t in data["added_tokens"]}
    assert flags["<|im_start|>"] is True and flags["<think>"] is False and flags["<|buffer1|>"] is False
    config = json.loads((tmp_path / "tokenizer_config.json").read_text(encoding="utf-8"))
    assert config["eos_token"] == "<|im_end|>"
    reloaded = Tokenizer.from_file(path)
    assert reloaded.encode("hello").ids == tiny_tokenizer().encode("hello").ids


def test_iter_corpus_reads_both_schemas(tmp_path):
    path = tmp_path / "mixed.jsonl"
    rows = [
        {"text": "plain document"},
        {"conversations": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]},
        {"text": ""},
        "not json",
    ]
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write((json.dumps(row) if not isinstance(row, str) else row) + "\n")
    assert list(iter_corpus([str(path)])) == ["plain document", "q\na"]
    assert list(iter_corpus([str(path)], max_docs_per_file=1)) == ["plain document"]
    assert list(iter_corpus([str(path), str(path)], max_docs_per_file=1)) == ["plain document"] * 2
