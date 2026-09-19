import math
from pathlib import Path

import pytest
import torch
from transformers import AutoTokenizer

from manas.config import ManasConfig
from manas.eval.benchmarks import hellaswag_records, labelled_records, mmlu_records
from manas.eval.multiple_choice import accuracy, choice_log_likelihood, predict, random_baseline
from manas.model.causal_lm import ManasForCausalLM

TOKENIZER_DIR = Path(__file__).resolve().parents[1] / "tokenizer"
CONFIG = ManasConfig(hidden_size=64, num_hidden_layers=2)


@pytest.fixture(scope="module")
def model_and_tokenizer():
    torch.manual_seed(0)
    return ManasForCausalLM(CONFIG).eval(), AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))


def test_mmlu_rows_become_scoreable_records():
    rows = [
        {"question": "What is 2+2?", "choices": ["3", "4", "5", "6"], "answer": 1},
        {"question": "broken", "choices": [], "answer": 0},
        {"question": "out of range", "choices": ["a"], "answer": 7},
    ]
    records = mmlu_records(rows)
    assert len(records) == 1
    assert records[0]["context"].endswith("Answer:")
    assert records[0]["choices"] == [" 3", " 4", " 5", " 6"] and records[0]["answer"] == 1


def test_arc_style_rows_resolve_the_answer_letter():
    rows = [
        {"question": "Which is a planet?", "choices": {"text": ["Mars", "Cheese"], "label": ["A", "B"]},
         "answerKey": "A"},
        {"question": "no key", "choices": {"text": ["x"], "label": ["A"]}, "answerKey": "Z"},
    ]
    records = labelled_records(rows)
    assert len(records) == 1 and records[0]["answer"] == 0
    stem_rows = [{"question_stem": "Ice is", "choices": {"text": ["cold", "hot"], "label": ["A", "B"]},
                  "answerKey": "B"}]
    assert labelled_records(stem_rows, question_key="question_stem")[0]["answer"] == 1


def test_hellaswag_rows_keep_the_context_and_endings():
    rows = [
        {"activity_label": "Baking", "ctx": "She cracks an egg and", "endings": ["stirs it", "drives away"],
         "label": "0"},
        {"activity_label": "x", "ctx": "y", "endings": ["a"], "label": "not a number"},
    ]
    records = hellaswag_records(rows)
    assert len(records) == 1
    assert records[0]["context"] == "Baking: She cracks an egg and"
    assert records[0]["answer"] == 0


def test_scores_are_length_normalised_log_probabilities(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    scores = choice_log_likelihood(model, tokenizer, "The capital of France is", [" Paris", " Madrid"])
    assert len(scores) == 2 and all(score < 0 for score in scores)
    raw = choice_log_likelihood(model, tokenizer, "The capital of France is", [" Paris"], normalize=False)
    assert raw[0] <= scores[0] or math.isclose(raw[0], scores[0])
    chosen, _ = predict(model, tokenizer, "The capital of France is", [" Paris", " Madrid"])
    assert chosen in (0, 1)


def test_a_model_that_memorised_one_answer_scores_it_highest(model_and_tokenizer):
    _, tokenizer = model_and_tokenizer
    torch.manual_seed(1)
    model = ManasForCausalLM(CONFIG)
    context, correct, wrong = "Question: colour of the sky?\nAnswer:", " blue", " banana"
    ids = tokenizer(context + correct, return_tensors="pt").input_ids
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for _ in range(60):
        loss = model(ids, labels=ids).loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()
    chosen, scores = predict(model, tokenizer, context, [correct, wrong])
    assert chosen == 0 and scores[0] > scores[1]


def test_accuracy_and_chance_baseline(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    records = [
        {"context": "Question: a?\nAnswer:", "choices": [" yes", " no"], "answer": 0},
        {"context": "Question: b?\nAnswer:", "choices": [" one", " two", " three", " four"], "answer": 2},
    ]
    score, seen = accuracy(records, model, tokenizer)
    assert 0.0 <= score <= 1.0 and seen == 2
    assert math.isclose(random_baseline(records), (0.5 + 0.25) / 2)
    assert random_baseline([]) == 0.0
    _, limited = accuracy(records, model, tokenizer, limit=1)
    assert limited == 1
