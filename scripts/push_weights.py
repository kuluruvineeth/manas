import argparse
import json
import os

from manas.training.metrics import read_metrics

STAGE_BLURB = {
    "pretrain": "next-token pretraining on English web, textbook, and story text",
    "full_sft": "supervised fine-tuning on English conversations, reasoning traces, and tool calls",
    "dpo": "preference alignment with DPO",
    "lora": "a LoRA adapter",
}

EXPORT_FILES = (
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
)


def canonical_stage(stage):
    return stage.removesuffix("_full")


BENCHMARK_CAVEAT = (
    "A 64M model has nowhere near the capacity these benchmarks assume, so scores sit at or "
    "near random chance. They are published because hiding them would be dishonest, not "
    "because they are good. What the model can actually do is hold a conversation, answer "
    "simple factual questions, and call tools."
)


def benchmark_section(benchmarks):
    if not benchmarks:
        return []
    lines = ["## Benchmarks", "", "| benchmark | accuracy | random chance | items |", "|---|---|---|---|"]
    for name, row in sorted(benchmarks["results"].items()):
        lines.append(
            f"| {name} | {row['accuracy']:.1%} | {row['chance']:.1%} | {row['items']} |"
        )
    lines += [f"| **average** | **{benchmarks['average']:.1%}** | | |", "", BENCHMARK_CAVEAT, ""]
    return lines


def model_card(stage, hidden_size, metrics_rows, extra_files, has_export=False, benchmarks=None):
    stage = canonical_stage(stage)
    last_train = next((r["loss"] for r in reversed(metrics_rows) if "loss" in r), None)
    last_val = next((r["val_loss"] for r in reversed(metrics_rows) if "val_loss" in r), None)
    lines = [
        "---",
        "language:",
        "- en",
        "license: apache-2.0",
        "pipeline_tag: text-generation",
        "tags:",
        "- manas",
        "- from-scratch",
        "---",
        "",
        f"# manas-64m · {stage}",
        "",
        f"Manas is a {hidden_size}-wide, 8-layer language model trained entirely from scratch. "
        f"This checkpoint is the **{stage}** stage: {STAGE_BLURB.get(stage, stage)}.",
        "",
        "## Numbers",
        "",
        f"- final training loss: {last_train:.4f}" if last_train is not None else "- final training loss: n/a",
        f"- final held-out loss: {last_val:.4f}" if last_val is not None else "- final held-out loss: n/a",
        "",
    ]
    lines += benchmark_section(benchmarks)
    lines += [
        "## Files",
        "",
    ]
    lines += [f"- `{name}`" for name in extra_files]
    lines += ["", "## Load", ""]
    if has_export:
        lines += [
            "The weights are exported in the Qwen3 layout, so nothing custom is needed:",
            "",
            "```python",
            "from transformers import AutoModelForCausalLM, AutoTokenizer",
            "",
            'model = AutoModelForCausalLM.from_pretrained("REPO_ID")',
            'tokenizer = AutoTokenizer.from_pretrained("REPO_ID")',
            "```",
            "",
            "It serves under vLLM the same way:",
            "",
            "```bash",
            "vllm serve REPO_ID --max-model-len 2048",
            "```",
            "",
            "The raw training checkpoint is also here for use with the training repo:",
            "",
        ]
    lines += [
        "```python",
        "from manas.config import ManasConfig",
        "from manas.training.utils import init_model",
        f'model, tokenizer = init_model(ManasConfig(hidden_size={hidden_size}), "{stage}", save_dir=".")',
        "```",
        "",
    ]
    return "\n".join(lines)


def collect_files(save_dir, stage, hidden_size, tokenizer_dir, export_dir=None, use_moe=False):
    stem = f"{stage}_{hidden_size}{'_moe' if use_moe else ''}"
    weight = os.path.join(save_dir, f"{stem}.pth")
    metrics = os.path.join(save_dir, f"{stem}_metrics.jsonl")
    curves = os.path.join(save_dir, f"{stem}_curves.png")
    files = {os.path.basename(weight): weight}
    for path in (metrics, curves):
        if os.path.exists(path):
            files[os.path.basename(path)] = path
    for name in ("tokenizer.json", "tokenizer_config.json"):
        files[name] = os.path.join(tokenizer_dir, name)
    if export_dir:
        for name in EXPORT_FILES:
            path = os.path.join(export_dir, name)
            if os.path.exists(path):
                files[name] = path
    return files


def push(
    stage,
    hidden_size,
    save_dir,
    tokenizer_dir,
    repo_id,
    private=True,
    dry_run=False,
    export_dir=None,
    use_moe=False,
    eval_path=None,
):
    files = collect_files(save_dir, stage, hidden_size, tokenizer_dir, export_dir, use_moe)
    missing = [p for p in files.values() if not os.path.exists(p)]
    if missing:
        raise SystemExit(f"missing files: {missing}")
    stem = f"{stage}_{hidden_size}{'_moe' if use_moe else ''}"
    metrics_path = files.get(f"{stem}_metrics.jsonl")
    rows = read_metrics(metrics_path) if metrics_path else []
    benchmarks = None
    if eval_path and os.path.exists(eval_path):
        with open(eval_path) as handle:
            benchmarks = json.load(handle)
    card = model_card(stage, hidden_size, rows, sorted(files), bool(export_dir), benchmarks)
    card = card.replace("REPO_ID", repo_id)
    if dry_run:
        print(f"[dry-run] would push to {repo_id} (private={private}):")
        for name, path in files.items():
            print(f"  {name} <- {path}")
        print(card)
        return
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
    api.upload_file(path_or_fileobj=card.encode("utf-8"), path_in_repo="README.md", repo_id=repo_id)
    config = {
        "stage": stage,
        "hidden_size": hidden_size,
        "num_hidden_layers": 8,
        "use_moe": use_moe,
        "architecture": "ManasForCausalLM",
    }
    api.upload_file(path_or_fileobj=json.dumps(config, indent=2).encode(), path_in_repo="manas.json", repo_id=repo_id)
    for name, path in files.items():
        print(f"[push] {name}", flush=True)
        api.upload_file(path_or_fileobj=path, path_in_repo=name, repo_id=repo_id)
    print(f"[done] https://huggingface.co/{repo_id}")


def main():
    parser = argparse.ArgumentParser(description="Publish a Manas checkpoint to the Hugging Face hub")
    parser.add_argument("--stage", default="pretrain")
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--save_dir", default="out")
    parser.add_argument("--tokenizer_dir", default="tokenizer")
    parser.add_argument("--repo_id", default=None)
    parser.add_argument("--export_dir", default=None, help="Qwen3 export from convert_model.py")
    parser.add_argument("--use_moe", type=int, default=0)
    parser.add_argument("--eval_path", default=None)
    parser.add_argument("--public", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    repo_id = args.repo_id or f"kuluruvineeth/manas-64m-{canonical_stage(args.stage).replace('_', '-')}"
    push(
        args.stage,
        args.hidden_size,
        args.save_dir,
        args.tokenizer_dir,
        repo_id,
        not args.public,
        args.dry_run,
        args.export_dir,
        bool(args.use_moe),
        args.eval_path,
    )


if __name__ == "__main__":
    main()
