import torch
from datasets import load_dataset
from torch.utils.data import Dataset

from manas.data.sft import assistant_labels, maybe_drop_empty_think


class DPODataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = load_dataset("json", data_files=data_path, split="train")
        self.start_ids = tokenizer(f"{tokenizer.bos_token}assistant\n", add_special_tokens=False).input_ids
        self.end_ids = tokenizer(f"{tokenizer.eos_token}\n", add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.samples)

    def encode(self, messages):
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        prompt = maybe_drop_empty_think(prompt)
        input_ids = self.tokenizer(prompt, truncation=True, max_length=self.max_length).input_ids
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        labels = assistant_labels(input_ids, self.start_ids, self.end_ids, self.max_length)
        mask = [0 if label == -100 else 1 for label in labels]
        ids = torch.tensor(input_ids, dtype=torch.long)
        return ids[:-1], ids[1:], torch.tensor(mask[1:], dtype=torch.long)

    def __getitem__(self, index):
        sample = self.samples[index]
        x_chosen, y_chosen, mask_chosen = self.encode(sample["chosen"])
        x_rejected, y_rejected, mask_rejected = self.encode(sample["rejected"])
        return {
            "x_chosen": x_chosen, "y_chosen": y_chosen, "mask_chosen": mask_chosen,
            "x_rejected": x_rejected, "y_rejected": y_rejected, "mask_rejected": mask_rejected,
        }
