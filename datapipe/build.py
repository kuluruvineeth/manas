import argparse
import json
import os
import random

from datapipe.schema import validate_line
from datapipe.sources import by_kind
from datapipe.synthesize import iter_agent_rows, iter_identity_rows
from datapipe.transforms import TRANSFORMS

DATASET_DIR = os.environ.get("MANAS_DATASET_DIR", "dataset")

PRETRAIN_BUDGET_BYTES = 9.5e9
PRETRAIN_MIN_CHARS = 200
PRETRAIN_MAX_CHARS = 60000
SFT_MIN_CHARS = 10
SFT_MAX_CHARS = 24000
DPO_MAX_SIDE_CHARS = 8000
MEDICAL_BUDGET_BYTES = 34e6
MEDICAL_MAX_CHARS = 6000
RLAIF_TARGET_ROWS = 30000
RLAIF_SAMPLE_PROB = 0.05
RLAIF_MAX_PROMPT_CHARS = 1200
MINI_TARGETS = {
    "pretrain_t2t.jsonl": ("pretrain_t2t_mini.jsonl", 1.2e9),
    "sft_t2t.jsonl": ("sft_t2t_mini.jsonl", 1.7e9),
}


class JsonlWriter:
    def __init__(self, name, out_dir=None, byte_budget=None):
        self.path = os.path.join(out_dir or DATASET_DIR, name)
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        self.tmp = self.path + ".tmp"
        self.file = open(self.tmp, "w", encoding="utf-8")  # noqa: SIM115
        self.byte_budget = byte_budget
        self.bytes = 0
        self.rows = 0

    def write(self, row):
        line = json.dumps(row, ensure_ascii=False) + "\n"
        self.write_line(line)

    def write_line(self, line):
        self.file.write(line)
        self.bytes += len(line.encode("utf-8"))
        self.rows += 1

    def full(self):
        return self.byte_budget is not None and self.bytes >= self.byte_budget

    def close(self):
        self.file.close()
        os.replace(self.tmp, self.path)
        print(f"[done] {self.path}: {self.rows:,} rows, {self.bytes/1e9:.2f} GB", flush=True)


def conversation_chars(row):
    return sum(
        len(message.get("content") or "") + len(message.get("reasoning_content") or "")
        for message in row["conversations"]
    )


def keep_sft(row):
    return SFT_MIN_CHARS <= conversation_chars(row) <= SFT_MAX_CHARS


def keep_dpo(row):
    return all(
        sum(len(m["content"]) for m in row[side]) <= DPO_MAX_SIDE_CHARS
        for side in ("chosen", "rejected")
    )


def iter_source(source):
    from datasets import load_dataset

    dataset = load_dataset(
        source.repo, name=source.config, split=source.split, streaming=source.streaming
    )
    transform = TRANSFORMS[source.transform]
    for raw in dataset:
        row = transform(raw)
        if row is not None:
            yield row


def emit(writer, row, kind):
    error = validate_line(kind, json.dumps(row, ensure_ascii=False))
    if error:
        raise ValueError(f"{kind} row failed schema: {error}")
    writer.write(row)


def build_pretrain(out_dir=None, budget=PRETRAIN_BUDGET_BYTES):
    writer = JsonlWriter("pretrain_t2t.jsonl", out_dir, byte_budget=budget)
    for source in by_kind("pretrain"):
        source_budget = writer.bytes + budget * source.weight
        print(f"[source] {source.repo} ({source.config}) -> +{budget*source.weight/1e9:.1f} GB", flush=True)
        for row in iter_source(source):
            text = row["text"]
            if len(text) < PRETRAIN_MIN_CHARS:
                continue
            if len(text) > PRETRAIN_MAX_CHARS:
                row = {"text": text[:PRETRAIN_MAX_CHARS]}
            emit(writer, row, "pretrain")
            if writer.bytes >= source_budget or writer.full():
                break
        if writer.full():
            break
    writer.close()


def build_sft(out_dir=None):
    writer = JsonlWriter("sft_t2t.jsonl", out_dir)
    for source in by_kind("sft"):
        print(f"[source] {source.repo} ({source.config})", flush=True)
        kept = 0
        for row in iter_source(source):
            if keep_sft(row):
                emit(writer, row, "sft")
                kept += 1
        print(f"[source done] {source.repo} ({source.config}): {kept:,} rows", flush=True)
    writer.close()


def build_dpo(out_dir=None):
    writer = JsonlWriter("dpo.jsonl", out_dir)
    for source in by_kind("dpo"):
        for row in iter_source(source):
            if keep_dpo(row):
                emit(writer, row, "dpo")
    writer.close()


def build_medical(out_dir=None):
    writer = JsonlWriter("lora_medical.jsonl", out_dir, byte_budget=MEDICAL_BUDGET_BYTES)
    for source in by_kind("medical"):
        for row in iter_source(source):
            if conversation_chars(row) > MEDICAL_MAX_CHARS:
                continue
            emit(writer, row, "sft")
            if writer.full():
                break
    writer.close()


def build_exam(out_dir=None):
    writer = JsonlWriter("lora_exam.jsonl", out_dir)
    for source in by_kind("exam"):
        print(f"[source] {source.repo} ({source.config}, {source.split})", flush=True)
        for row in iter_source(source):
            emit(writer, row, "sft")
    writer.close()


def build_identity(out_dir=None):
    writer = JsonlWriter("lora_identity.jsonl", out_dir)
    for row in iter_identity_rows():
        emit(writer, row, "sft")
    writer.close()


def build_agent(out_dir=None):
    writer = JsonlWriter("agent_rl.jsonl", out_dir)
    for row in iter_agent_rows():
        emit(writer, row, "agent")
    writer.close()
    math_writer = JsonlWriter("agent_rl_math.jsonl", out_dir)
    for source in by_kind("math"):
        for row in iter_source(source):
            emit(math_writer, row, "agent")
    math_writer.close()


def is_plain_chat(conversation):
    return not any(
        message.get("tools") or message.get("tool_calls") or message["role"] == "tool"
        for message in conversation
    )


def build_rlaif(out_dir=None, target_rows=RLAIF_TARGET_ROWS, seed=42):
    rng = random.Random(seed)
    out_dir = out_dir or DATASET_DIR
    writer = JsonlWriter("rlaif.jsonl", out_dir)
    with open(os.path.join(out_dir, "sft_t2t.jsonl"), encoding="utf-8") as source_file:
        for line in source_file:
            if writer.rows >= target_rows:
                break
            if rng.random() > RLAIF_SAMPLE_PROB:
                continue
            conversation = json.loads(line)["conversations"]
            if not is_plain_chat(conversation):
                continue
            user_indexes = [i for i, m in enumerate(conversation) if m["role"] == "user"]
            if not user_indexes:
                continue
            prompt = [
                {"role": m["role"], "content": m["content"]}
                for m in conversation[: user_indexes[-1] + 1]
            ]
            if not 10 <= len(prompt[-1]["content"]) <= RLAIF_MAX_PROMPT_CHARS:
                continue
            prompt.append({"role": "assistant", "content": "none"})
            emit(writer, {"conversations": prompt}, "rlaif")
    writer.close()


def build_minis(out_dir=None, seed=42):
    rng = random.Random(seed)
    out_dir = out_dir or DATASET_DIR
    for source_name, (mini_name, target_bytes) in MINI_TARGETS.items():
        source_path = os.path.join(out_dir, source_name)
        probability = min(1.0, target_bytes / os.path.getsize(source_path))
        writer = JsonlWriter(mini_name, out_dir, byte_budget=target_bytes)
        with open(source_path, encoding="utf-8") as source_file:
            for line in source_file:
                if rng.random() <= probability:
                    writer.write_line(line)
                    if writer.full():
                        break
        writer.close()


BUILDERS = {
    "pretrain": build_pretrain,
    "sft": build_sft,
    "dpo": build_dpo,
    "medical": build_medical,
    "exam": build_exam,
    "identity": build_identity,
    "agent": build_agent,
    "rlaif": build_rlaif,
    "minis": build_minis,
}

ORDER = ["pretrain", "sft", "dpo", "medical", "exam", "identity", "agent", "rlaif", "minis"]


def main():
    parser = argparse.ArgumentParser(description="Build the Manas dataset bundle")
    parser.add_argument("targets", nargs="+", choices=[*ORDER, "all"])
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()
    targets = ORDER if "all" in args.targets else args.targets
    for target in targets:
        print(f"[build] {target}", flush=True)
        BUILDERS[target](args.out_dir)


if __name__ == "__main__":
    main()
