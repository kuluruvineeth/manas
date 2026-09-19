import json

from torch.utils.data import Dataset


class AgentRLDataset(Dataset):
    def __init__(self, data_path, max_rows=None):
        super().__init__()
        self.rows = []
        with open(data_path, encoding="utf-8") as f:
            for line in f:
                if max_rows is not None and len(self.rows) >= max_rows:
                    break
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                messages = list(row["conversations"])[:-1]
                tools = None
                for message in messages:
                    if message.get("role") == "system" and message.get("tools"):
                        raw = message["tools"]
                        tools = json.loads(raw) if isinstance(raw, str) else raw
                clean = [{"role": m["role"], "content": m["content"]} for m in messages]
                self.rows.append({"messages": clean, "tools": tools or [], "gt": [str(g) for g in row.get("gt", [])]})

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def collate_agent(batch):
    return list(batch)
