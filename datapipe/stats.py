import argparse
import json
import os
import random

from datapipe.build import DATASET_DIR
from datapipe.schema import CHECKERS

FILES = {
    "pretrain_t2t.jsonl": "pretrain",
    "pretrain_t2t_mini.jsonl": "pretrain",
    "sft_t2t.jsonl": "sft",
    "sft_t2t_mini.jsonl": "sft",
    "dpo.jsonl": "dpo",
    "rlaif.jsonl": "rlaif",
    "agent_rl.jsonl": "agent",
    "agent_rl_math.jsonl": "agent",
    "lora_identity.jsonl": "sft",
    "lora_medical.jsonl": "sft",
    "lora_exam.jsonl": "sft",
}

CHARS_PER_TOKEN = 4.3


def sample_lines(path, sample_size, seed=0):
    rng = random.Random(seed)
    size = os.path.getsize(path)
    lines = []
    with open(path, "rb") as f:
        while len(lines) < sample_size:
            f.seek(rng.randint(0, max(0, size - 2)))
            f.readline()
            line = f.readline()
            if not line:
                f.seek(0)
                line = f.readline()
            lines.append(line.decode("utf-8", errors="replace"))
    return lines


def row_chars(row):
    if "text" in row:
        return len(row["text"])
    if "chosen" in row:
        return sum(len(m["content"]) for side in ("chosen", "rejected") for m in row[side])
    return sum(
        len(m.get("content") or "") + len(m.get("reasoning_content") or "")
        for m in row.get("conversations", [])
    )


def file_stats(path, kind, sample_size):
    checker = CHECKERS[kind]
    lines = sample_lines(path, sample_size)
    errors = 0
    chars = []
    tool_rows = 0
    think_rows = 0
    for line in lines:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            errors += 1
            continue
        if checker(row):
            errors += 1
            continue
        chars.append(row_chars(row))
        messages = row.get("conversations") or []
        if any(m.get("tools") or m.get("tool_calls") for m in messages):
            tool_rows += 1
        if any(m.get("reasoning_content") for m in messages):
            think_rows += 1
    size_bytes = os.path.getsize(path)
    avg_line_bytes = sum(len(line.encode("utf-8")) for line in lines) / len(lines)
    return {
        "size_gb": size_bytes / 1e9,
        "est_rows": int(size_bytes / avg_line_bytes),
        "est_tokens_m": size_bytes / CHARS_PER_TOKEN / 1e6,
        "sampled": len(lines),
        "errors": errors,
        "avg_chars": sum(chars) / len(chars) if chars else 0,
        "tool_pct": 100 * tool_rows / len(lines),
        "think_pct": 100 * think_rows / len(lines),
    }


def report(dataset_dir=None, sample_size=1000):
    dataset_dir = dataset_dir or DATASET_DIR
    header = (
        f"| {'file':26s} | {'size':>8s} | {'~rows':>10s} | {'~tokens':>9s} "
        f"| {'errors':>6s} | {'avg chars':>9s} | {'tool%':>5s} | {'think%':>6s} |"
    )
    divider = "|" + "|".join("-" * (len(part) - 1) for part in header.split("|")[1:-1]) + "|"
    lines = [header, divider]
    failed = False
    for name, kind in FILES.items():
        path = os.path.join(dataset_dir, name)
        if not os.path.exists(path):
            lines.append(f"| {name:26s} | {'MISSING':>8s} |")
            failed = True
            continue
        stats = file_stats(path, kind, sample_size)
        if stats["errors"]:
            failed = True
        lines.append(
            f"| {name:26s} | {stats['size_gb']:7.2f}G | {stats['est_rows']:>10,} "
            f"| {stats['est_tokens_m']:8.0f}M | {stats['errors']:>6d} "
            f"| {stats['avg_chars']:>9.0f} | {stats['tool_pct']:5.1f} | {stats['think_pct']:6.1f} |"
        )
    return "\n".join(lines), failed


def main():
    parser = argparse.ArgumentParser(description="Report Manas dataset bundle health")
    parser.add_argument("--dataset-dir", default=None)
    parser.add_argument("--sample", type=int, default=1000)
    args = parser.parse_args()
    text, failed = report(args.dataset_dir, args.sample)
    print(text)
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
