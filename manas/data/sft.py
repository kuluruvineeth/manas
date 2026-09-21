import json
import random

import torch
from datasets import Features, Value, load_dataset
from torch.utils.data import Dataset

from manas.tokenizer import EMPTY_THINK

SYSTEM_PROMPTS = [
    "You are Manas, a small but useful language model.",
    "You are a helpful AI assistant.",
    "You are Manas, a lightweight intelligent assistant.",
    "You are a friendly chatbot. Please answer the user's questions carefully.",
    "You are a knowledgeable AI. Try your best to provide accurate information.",
    "You are Manas. Do your best to help the user solve their problem.",
    "You are a reliable AI assistant. Give accurate, well-reasoned answers.",
    "You are Manas, a compact language model trained from scratch.",
]


CONVERSATION_FEATURES = Features({
    "conversations": [{
        "role": Value("string"),
        "content": Value("string"),
        "reasoning_content": Value("string"),
        "tools": Value("string"),
        "tool_calls": Value("string"),
    }]
})


def maybe_add_system_prompt(conversations, add_system_ratio=0.2, rng=random):
    if any(message.get("tools") for message in conversations):
        return conversations
    if conversations[0].get("role") != "system" and rng.random() < add_system_ratio:
        return [{"role": "system", "content": rng.choice(SYSTEM_PROMPTS)}] + conversations
    return conversations


def maybe_drop_empty_think(prompt, empty_think_ratio=0.2, rng=random):
    if EMPTY_THINK in prompt and rng.random() > empty_think_ratio:
        return prompt.replace(EMPTY_THINK, "")
    return prompt


def complete_tool_call(call):
    call = dict(call.get("function", call))
    arguments = call.get("arguments", call.get("parameters", {}))
    return {"name": call.get("name", ""), "arguments": arguments if arguments is not None else {}}


def render_conversation(tokenizer, conversations):
    messages = []
    tools = None
    for message in conversations:
        message = {k: v for k, v in dict(message).items() if v is not None}
        if message.get("role") == "system" and message.get("tools"):
            tools = json.loads(message["tools"]) if isinstance(message["tools"], str) else message["tools"]
        if isinstance(message.get("tool_calls"), str):
            message["tool_calls"] = json.loads(message["tool_calls"])
        if message.get("tool_calls"):
            message["tool_calls"] = [complete_tool_call(call) for call in message["tool_calls"]]
        messages.append(message)
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, tools=tools)


def assistant_labels(input_ids, start_ids, end_ids, max_length):
    labels = [-100] * len(input_ids)
    i = 0
    while i < len(input_ids):
        if input_ids[i : i + len(start_ids)] == start_ids:
            start = i + len(start_ids)
            end = start
            while end < len(input_ids) and input_ids[end : end + len(end_ids)] != end_ids:
                end += 1
            for j in range(start, min(end + len(end_ids), max_length)):
                labels[j] = input_ids[j]
            i = end + len(end_ids) if end < len(input_ids) else len(input_ids)
        else:
            i += 1
    return labels


class SFTDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = load_dataset("json", data_files=data_path, split="train", features=CONVERSATION_FEATURES)
        self.start_ids = tokenizer(f"{tokenizer.bos_token}assistant\n", add_special_tokens=False).input_ids
        self.end_ids = tokenizer(f"{tokenizer.eos_token}\n", add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        conversations = maybe_add_system_prompt(self.samples[index]["conversations"])
        prompt = maybe_drop_empty_think(render_conversation(self.tokenizer, conversations))
        input_ids = self.tokenizer(prompt).input_ids[: self.max_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        labels = assistant_labels(input_ids, self.start_ids, self.end_ids, self.max_length)
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)
