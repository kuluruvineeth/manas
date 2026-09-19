import json
from pathlib import Path

import pytest
from transformers import AutoTokenizer

from manas.tools.registry import tools_for
from manas.training.agent import (
    agent_reward,
    final_answer,
    numbers_in,
    observation_delta,
    unbalanced_tags,
    verify_ground_truth,
)

TOKENIZER_DIR = Path(__file__).resolve().parents[1] / "tokenizer"


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))


def call_block(name, arguments):
    return f'<tool_call>\n{json.dumps({"name": name, "arguments": arguments})}\n</tool_call>'


def test_numbers_are_extracted_without_swallowing_words():
    assert numbers_in("the answer is 161") == [161.0]
    assert numbers_in("1,234 apples and 5.5 pears") == [1234.0, 5.5]
    assert numbers_in("version2 of file3") == []


def test_ground_truth_matches_numerically_and_textually():
    assert verify_ground_truth("The result is 161.", ["161"]) == ["161"]
    assert verify_ground_truth("It is 161.0 exactly", ["161"]) == ["161"]
    assert verify_ground_truth("It is cloudy in London", ["cloudy"]) == ["cloudy"]
    assert verify_ground_truth("The result is 160", ["161"]) == []
    assert verify_ground_truth("14C, cloudy", ["14", "cloudy"]) == ["14", "cloudy"]


def test_tag_balance_and_final_answer():
    assert unbalanced_tags("<tool_call>x</tool_call>") == 0
    assert unbalanced_tags("<tool_call>x") == 1
    text = f"thinking {call_block('calculate_math', {'expression': '23*7'})} the answer is 161"
    assert final_answer(text) == "the answer is 161"
    assert final_answer("no tools, just words") == "no tools, just words"


def test_a_correct_tool_trajectory_scores_well():
    text = f"{call_block('calculate_math', {'expression': '23*7'})}\nThe answer is 161."
    reward = agent_reward(text, ["161"], {"calculate_math"}, unfinished=False)
    assert reward == pytest.approx(0.5 + 2.5)


def test_a_wrong_answer_loses_the_ground_truth_bonus():
    text = f"{call_block('calculate_math', {'expression': '23*7'})}\nThe answer is 200."
    assert agent_reward(text, ["161"], {"calculate_math"}, unfinished=False) == pytest.approx(0.5)


def test_calling_a_tool_that_was_not_offered_is_penalised():
    text = f"{call_block('get_current_time', {'timezone': 'Europe/London'})}\n14:30"
    offered_wrong = agent_reward(text, ["14:30"], {"calculate_math"}, unfinished=False)
    offered_right = agent_reward(text, ["14:30"], {"get_current_time"}, unfinished=False)
    assert offered_right > offered_wrong


def test_unfinished_trajectories_lose_the_answer_and_take_a_penalty():
    text = f"{call_block('calculate_math', {'expression': '23*7'})}\nThe answer is 161."
    finished = agent_reward(text, ["161"], {"calculate_math"}, unfinished=False)
    unfinished = agent_reward(text, ["161"], {"calculate_math"}, unfinished=True)
    assert unfinished == pytest.approx(finished - 2.5 - 0.5)


def test_unbalanced_tags_are_punished_and_reward_is_clamped():
    broken = "<tool_call>" * 12
    assert agent_reward(broken, ["161"], {"calculate_math"}, unfinished=False) == -3.0
    perfect = f"{call_block('calculate_math', {'expression': '1+1'})}\n2"
    assert agent_reward(perfect, ["2"], {"calculate_math"}, unfinished=False) <= 3.0


def test_no_tool_call_falls_back_to_the_shaped_reward():
    plain = "I think the answer is probably 161 but I did not check."
    assert agent_reward(plain, ["161"], {"calculate_math"}, unfinished=False) == -0.5
    assert agent_reward(plain, ["161"], {"calculate_math"}, unfinished=False, fallback_reward=1.25) == 1.25


def test_observation_delta_returns_only_the_spliced_tokens(tokenizer):
    tools = tools_for(["calculate_math"])
    assistant = {"role": "assistant", "content": call_block("calculate_math", {"expression": "23*7"})}
    messages = [
        {"role": "system", "content": "You are Manas."},
        {"role": "user", "content": "What is 23 times 7?"},
        assistant,
        {"role": "tool", "content": "161"},
    ]
    delta = observation_delta(
        tokenizer, messages, assistant, tools, open_thinking=False,
        add_generation_prompt=True, eos_id=tokenizer.eos_token_id,
    )
    text = tokenizer.decode(delta)
    assert "<tool_response>" in text and "161" in text
    assert "23 times 7" not in text
    assert text.rstrip().endswith("<think>\n\n</think>") or "assistant" in text
    assert assistant["content"] == call_block("calculate_math", {"expression": "23*7"})


def test_observation_delta_drops_a_duplicated_eos(tokenizer):
    tools = tools_for(["calculate_math"])
    assistant = {"role": "assistant", "content": "done"}
    messages = [{"role": "user", "content": "hi"}, assistant, {"role": "tool", "content": "ok"}]
    with_eos = observation_delta(
        tokenizer, messages, assistant, tools, False, True, tokenizer.eos_token_id,
        last_token_id=tokenizer.eos_token_id,
    )
    without = observation_delta(
        tokenizer, messages, assistant, tools, False, True, tokenizer.eos_token_id, last_token_id=None
    )
    assert len(with_eos) == len(without) - 1
    assert with_eos[0] != tokenizer.eos_token_id
