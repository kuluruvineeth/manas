import random

from datasets import load_dataset
from torch.utils.data import Dataset

from manas.data.sft import maybe_add_system_prompt


class RLAIFDataset(Dataset):
    def __init__(self, data_path, tokenizer, thinking_ratio=0.5, seed=42):
        super().__init__()
        self.tokenizer = tokenizer
        self.thinking_ratio = thinking_ratio
        self.rng = random.Random(seed)
        self.samples = load_dataset("json", data_files=data_path, split="train")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        conversations = list(self.samples[index]["conversations"])[:-1]
        conversations = maybe_add_system_prompt(conversations, rng=self.rng)
        prompt = self.tokenizer.apply_chat_template(
            conversations,
            tokenize=False,
            add_generation_prompt=True,
            open_thinking=self.rng.random() < self.thinking_ratio,
        )
        return {"prompt": prompt}


def collate_prompts(batch):
    return [row["prompt"] for row in batch]


def tokenize_prompts(tokenizer, prompts, max_length, device="cpu"):
    tokenizer.padding_side = "left"
    encoded = tokenizer(
        prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_length
    )
    return encoded.input_ids.to(device), encoded.attention_mask.to(device)
