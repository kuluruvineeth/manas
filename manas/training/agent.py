import re

import torch

from manas.tools.registry import execute, is_valid_call, parse_tool_calls

NUMBER = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?![\w.])")
GT_HIT_REWARD = 2.5
TOOL_ALIGNMENT_REWARD = 0.5
UNFINISHED_PENALTY = 0.5
TAG_PENALTY = 0.5
REWARD_CLAMP = 3.0


def numbers_in(text):
    return [float(match) for match in NUMBER.findall(text.replace(",", ""))]


def verify_ground_truth(text, ground_truth):
    lowered = text.lower()
    found = numbers_in(text)
    verified = []
    for item in ground_truth:
        target = str(item).strip()
        if target.lower() in lowered:
            verified.append(target)
            continue
        try:
            wanted = float(target.replace(",", ""))
        except ValueError:
            continue
        if any(abs(wanted - value) < 1e-6 for value in found):
            verified.append(target)
    return verified


def unbalanced_tags(text):
    return abs(text.count("<tool_call>") - text.count("</tool_call>"))


def final_answer(text):
    _, marker, tail = text.rpartition("</tool_call>")
    return tail.strip() if marker else text.strip()


def clamp_reward(value):
    return max(min(value, REWARD_CLAMP), -REWARD_CLAMP)


def agent_reward(trajectory_text, ground_truth, offered_names, unfinished, fallback_reward=None):
    reward = -TAG_PENALTY * unbalanced_tags(trajectory_text)
    calls = parse_tool_calls(trajectory_text)
    if not calls:
        no_call = fallback_reward if fallback_reward is not None else -TOOL_ALIGNMENT_REWARD
        return clamp_reward(reward + no_call)

    valid = [call for call in calls if is_valid_call(call, offered_names)]
    gap = abs(len(valid) - len(ground_truth)) + max(0, len(calls) - len(valid))
    reward += TOOL_ALIGNMENT_REWARD if gap == 0 else -TOOL_ALIGNMENT_REWARD * gap

    answer = "" if unfinished else final_answer(trajectory_text)
    if ground_truth:
        verified = verify_ground_truth(answer, ground_truth)
        reward += GT_HIT_REWARD * len(verified) / len(ground_truth)
    if unfinished:
        reward -= UNFINISHED_PENALTY
    return clamp_reward(reward)


def observation_delta(tokenizer, messages, assistant_message, tools, open_thinking, add_generation_prompt, eos_id,
                      last_token_id=None):
    """Tokens the chat template inserts for a tool observation, found by splicing on a unique marker."""
    marker = f"<|agent_observation_{id(messages)}_{len(messages)}|>"
    original = assistant_message["content"]
    assistant_message["content"] = original + marker
    try:
        marked = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt,
            tools=tools, open_thinking=open_thinking,
        )
    finally:
        assistant_message["content"] = original
    _, found, observation = marked.partition(marker)
    if not found:
        raise RuntimeError("chat template did not preserve the assistant content boundary")
    delta = tokenizer(observation, add_special_tokens=False).input_ids
    if last_token_id == eos_id and delta[:1] == [eos_id]:
        delta = delta[1:]
    return delta


def rollout_trajectory(model, tokenizer, messages, tools, max_turns=3, max_new_tokens=256, temperature=0.8,
                       open_thinking=False, device="cpu"):
    offered = {tool["function"]["name"] for tool in tools}
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, tools=tools, open_thinking=open_thinking
    )
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
    conversation = list(messages)
    response_ids, response_mask, texts = [], [], []
    unfinished = False

    for turn in range(max_turns):
        context = torch.tensor([prompt_ids + response_ids], device=device)
        with torch.no_grad():
            generated = model.generate(
                context, max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature,
                top_k=0, top_p=1.0, eos_token_id=tokenizer.eos_token_id,
            )
        new_ids = generated[0, context.size(1):].tolist()
        text = tokenizer.decode(new_ids, skip_special_tokens=True)
        response_ids.extend(new_ids)
        response_mask.extend([1] * len(new_ids))
        texts.append(text)

        calls = parse_tool_calls(text)
        if not calls:
            break
        if turn == max_turns - 1:
            unfinished = True
            break

        assistant_message = {"role": "assistant", "content": text}
        conversation = [*conversation, assistant_message]
        for call in calls:
            conversation.append({"role": "tool", "content": execute(call)})
        delta = observation_delta(
            tokenizer, conversation, assistant_message, tools, open_thinking,
            add_generation_prompt=True, eos_id=tokenizer.eos_token_id,
            last_token_id=new_ids[-1] if new_ids else None,
        )
        response_ids.extend(delta)
        response_mask.extend([0] * len(delta))
        texts.append(tokenizer.decode(delta, skip_special_tokens=True))

    return {
        "prompt_ids": prompt_ids,
        "response_ids": response_ids,
        "response_mask": response_mask,
        "text": "".join(texts),
        "unfinished": unfinished,
        "offered": offered,
    }
