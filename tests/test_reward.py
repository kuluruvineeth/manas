import torch

from manas.training.reward import calculate_rewards, format_reward, parse_prompt, repetition_penalty


class StubRewardModel:
    def __init__(self, score=1.0):
        self.score = score
        self.calls = []

    def get_score(self, messages, response):
        self.calls.append((messages, response))
        return self.score


def test_parse_prompt_recovers_the_conversation():
    prompt = (
        "<|im_start|>system\nBe brief.<|im_end|>\n"
        "<|im_start|>user\nWhat is rain?<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    assert parse_prompt(prompt) == [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "What is rain?"},
    ]


def test_repetition_penalty_punishes_loops_and_is_capped():
    assert repetition_penalty("the cat sat on the mat quietly") == 0.0
    looped = "buy now " * 40
    assert repetition_penalty(looped) == 0.5
    assert repetition_penalty("too short") == 0.0


GOOD_ANSWER = (
    "Rain begins high inside a cloud, where countless tiny droplets drift about and collide. "
    "Each collision makes them slightly larger until gravity finally wins and they tumble earthward."
)
GOOD_THINKING = (
    "The child needs the idea of water gathering in the sky before anything about weight or gravity, "
    "so I will start with clouds and build up slowly toward why droplets eventually fall down."
)


def test_format_reward_rewards_a_reasonable_answer():
    reward, answer = format_reward(GOOD_ANSWER)
    assert reward == 0.5 and answer == GOOD_ANSWER

    reward, _ = format_reward("no")
    assert reward == -0.5


def test_format_reward_handles_thinking_blocks():
    response = f"<think>\n{GOOD_THINKING}\n</think>\n\n{GOOD_ANSWER}"
    reward, answer = format_reward(response)
    assert reward == 0.5 + 1.0 + 0.25
    assert answer == GOOD_ANSWER

    doubled = response + "</think>"
    assert format_reward(doubled)[0] < reward

    tiny_thinking = f"<think>\nhm\n</think>\n\n{GOOD_ANSWER}"
    assert format_reward(tiny_thinking)[0] == 0.5 - 0.5 + 0.25


def test_calculate_rewards_adds_the_model_score():
    prompt = "<|im_start|>user\nWhat is rain?<|im_end|>\n<|im_start|>assistant\n"
    response = GOOD_ANSWER
    without = calculate_rewards([prompt], [response])
    model = StubRewardModel(score=1.5)
    with_model = calculate_rewards([prompt], [response], reward_model=model)
    torch.testing.assert_close(with_model - without, torch.tensor([1.5]))
    assert model.calls[0][0][0]["content"] == "What is rain?"


def test_rewards_rank_a_good_answer_above_a_repetitive_one():
    prompt = "<|im_start|>user\nExplain rain.<|im_end|>\n<|im_start|>assistant\n"
    looping = "rain is rain is " * 30
    rewards = calculate_rewards([prompt, prompt], [GOOD_ANSWER, looping])
    assert rewards[0] > rewards[1]
