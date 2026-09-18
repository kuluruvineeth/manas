import argparse
import os

from datapipe.build import DATASET_DIR
from datapipe.sources import SOURCES
from datapipe.stats import FILES, report
from datapipe.synthesize import GENERATORS

REPO_ID = "kuluruvineeth/manas_dataset"

FILE_NOTES = {
    "pretrain_t2t.jsonl": "pretraining corpus, ~2.2B tokens",
    "pretrain_t2t_mini.jsonl": "pretraining corpus, quick-start tier",
    "sft_t2t.jsonl": "supervised fine-tuning conversations (tool-calling and reasoning mixed in)",
    "sft_t2t_mini.jsonl": "supervised fine-tuning, quick-start tier",
    "dpo.jsonl": "preference pairs for DPO",
    "rlaif.jsonl": "prompt-only rows for PPO/GRPO",
    "agent_rl.jsonl": "synthesized tool-use RL tasks with verifiable ground truth",
    "agent_rl_math.jsonl": "GSM8K word problems with numeric ground truth",
    "lora_identity.jsonl": "Manas identity adapter data",
    "lora_medical.jsonl": "medical-domain adapter data",
    "lora_exam.jsonl": "exam-format alignment data",
}


def build_card():
    lines = [
        "---",
        "language:",
        "- en",
        "license: other",
        "license_name: mixed-source",
        "license_link: https://huggingface.co/datasets/kuluruvineeth/manas_dataset#sources",
        "task_categories:",
        "- text-generation",
        "---",
        "",
        "# Manas Dataset",
        "",
        "The complete English training bundle for [Manas](https://github.com/kuluruvineeth), "
        "a lightweight language model trained entirely from scratch. Every file is rebuildable "
        "from raw sources with `python -m datapipe.build all`.",
        "",
        "## Files",
        "",
        "| file | stage |",
        "|---|---|",
    ]
    for name in FILES:
        lines.append(f"| `{name}` | {FILE_NOTES[name]} |")
    lines += [
        "",
        "## Sources",
        "",
        "Every upstream source, its license as stated on its dataset card, and why it was chosen:",
        "",
        "| source | license | why |",
        "|---|---|---|",
    ]
    for source in SOURCES:
        config = f" ({source.config})" if source.config else ""
        lines.append(f"| `{source.repo}`{config} | {source.license} | {source.why} |")
    lines += [
        "",
        f"`agent_rl.jsonl` is synthesized by {len(GENERATORS)} deterministic generators over "
        "fixed mock-tool tables, so every ground-truth answer is correct by construction.",
        "",
        "No source uses a benchmark test split.",
        "",
    ]
    return "\n".join(lines)


def push(repo_id=REPO_ID, dataset_dir=None, private=True, dry_run=False):
    dataset_dir = dataset_dir or DATASET_DIR
    text, failed = report(dataset_dir, sample_size=500)
    print(text, flush=True)
    if failed:
        raise SystemExit("bundle failed validation; not pushing")
    card = build_card()
    plan = [("README.md", "<generated card>")] + [
        (name, os.path.join(dataset_dir, name)) for name in FILES
    ]
    if dry_run:
        print(f"[dry-run] would push to {repo_id} (private={private}):", flush=True)
        for remote, local in plan:
            print(f"  {remote} <- {local}", flush=True)
        return
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    api.upload_file(
        path_or_fileobj=card.encode("utf-8"), path_in_repo="README.md",
        repo_id=repo_id, repo_type="dataset",
    )
    for name in FILES:
        print(f"[push] {name}", flush=True)
        api.upload_file(
            path_or_fileobj=os.path.join(dataset_dir, name), path_in_repo=name,
            repo_id=repo_id, repo_type="dataset",
        )
    print(f"[done] https://huggingface.co/datasets/{repo_id}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Publish the Manas dataset bundle to the HF hub")
    parser.add_argument("--repo-id", default=REPO_ID)
    parser.add_argument("--dataset-dir", default=None)
    parser.add_argument("--public", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    push(args.repo_id, args.dataset_dir, private=not args.public, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
