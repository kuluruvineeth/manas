import re

import torch

IM_TURN = re.compile(r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>", re.DOTALL)

# Upstream's bands are character counts tuned for Chinese (~1.6 chars per token).
# English runs ~4.3 chars per token, so the same token budget is roughly 2.5x the characters.
ANSWER_CHARS = (50, 2000)
THINKING_CHARS = (50, 750)
SCORE_CLAMP = 3.0


def parse_prompt(prompt):
    return [{"role": role, "content": content.strip()} for role, content in IM_TURN.findall(prompt)]


def repetition_penalty(text, n=3, cap=0.5):
    words = text.split()
    if len(words) < n + 1:
        return 0.0
    grams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    duplicates = len(grams) - len(set(grams))
    return min(duplicates / len(grams), cap)


def format_reward(response):
    reward = 0.5 if ANSWER_CHARS[0] <= len(response.strip()) <= ANSWER_CHARS[1] else -0.5
    answer = response
    if "</think>" in response:
        thinking, answer = response.split("</think>", 1)
        reward += 1.0 if THINKING_CHARS[0] <= len(thinking.strip()) <= THINKING_CHARS[1] else -0.5
        reward += 0.25 if response.count("</think>") == 1 else -0.25
        answer = answer.strip()
    return reward - repetition_penalty(answer), answer


class RewardModel:
    def __init__(self, model_path, device="cpu", dtype=torch.float32):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path, dtype=dtype, num_labels=1
        ).to(device).eval()
        self.device = device

    @torch.no_grad()
    def get_score(self, messages, response):
        conversation = [*messages, {"role": "assistant", "content": response}]
        text = self.tokenizer.apply_chat_template(conversation, tokenize=False)
        tokens = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=4096).to(self.device)
        score = self.model(**tokens).logits[0][0].item()
        return max(min(score, SCORE_CLAMP), -SCORE_CLAMP)


def calculate_rewards(prompts, responses, reward_model=None, device="cpu"):
    rewards = torch.zeros(len(responses), device=device)
    for index, (prompt, response) in enumerate(zip(prompts, responses, strict=True)):
        shaped, answer = format_reward(response)
        rewards[index] += shaped
        if reward_model is not None:
            rewards[index] += reward_model.get_score(parse_prompt(prompt), answer)
    return rewards
