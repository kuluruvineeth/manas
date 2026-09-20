import os
import subprocess
import sys
import time
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parents[1]
REMOTE_REPO = "/root/manas"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("torch", "transformers", "datasets", "tokenizers", "jinja2", "numpy", "wandb")
    .add_local_dir(
        str(REPO),
        remote_path=REMOTE_REPO,
        copy=True,
        ignore=[".venv", ".git", "dataset", "out", "checkpoints", "__pycache__", ".ruff_cache", ".pytest_cache"],
    )
)

app = modal.App("manas-train", image=image)
data_volume = modal.Volume.from_name("manas-data", create_if_missing=True)
out_volume = modal.Volume.from_name("manas-out", create_if_missing=True)
cache_volume = modal.Volume.from_name("manas-hf-cache", create_if_missing=True)
secrets = [modal.Secret.from_name("wandb-secret")]

VOLUMES = {"/data": data_volume, "/out": out_volume, "/cache": cache_volume}
CACHE_ENV = {"HF_HOME": "/cache/huggingface"}

STAGE_DATA = {
    "pretrain": "pretrain_t2t_mini.jsonl",
    "pretrain_full": "pretrain_t2t.jsonl",
    "full_sft": "sft_t2t_mini.jsonl",
    "full_sft_full": "sft_t2t.jsonl",
    "lora": "lora_identity.jsonl",
    "lora_mixed": "lora_identity_mixed.jsonl",
    "dpo": "dpo.jsonl",
    "distillation": "sft_t2t_mini.jsonl",
    "ppo": "rlaif.jsonl",
    "grpo": "rlaif.jsonl",
    "agent": "agent_rl.jsonl",
}

STAGE_TRAINER = {
    "lora_mixed": "lora",
}


def run_stage(stage, extra_args):
    trainer = STAGE_TRAINER.get(stage, stage.removesuffix("_full"))
    command = [
        sys.executable, f"{REMOTE_REPO}/trainer/train_{trainer}.py",
        "--data_path", f"/data/{STAGE_DATA[stage]}",
        "--tokenizer_dir", f"{REMOTE_REPO}/tokenizer",
        "--save_dir", "/out",
        "--device", "cuda:0",
        *extra_args,
    ]
    print("running:", " ".join(command), flush=True)
    env = {**os.environ, "PYTHONPATH": REMOTE_REPO, **CACHE_ENV}
    result = subprocess.run(command, cwd=REMOTE_REPO, env=env)
    out_volume.commit()
    cache_volume.commit()
    return result.returncode


@app.function(volumes=VOLUMES)
def check_inputs(stage: str, from_weight: str | None):
    missing = []
    if not Path(f"/data/{STAGE_DATA[stage]}").exists():
        missing.append(f"/data/{STAGE_DATA[stage]}")
    if from_weight and not list(Path("/out").glob(f"{from_weight}_*.pth")):
        missing.append(f"/out/{from_weight}_*.pth")
    if missing:
        have = sorted(p.name for p in Path("/data").iterdir()) + sorted(p.name for p in Path("/out").glob("*.pth"))
        return f"missing: {', '.join(missing)}\nvolumes hold: {', '.join(have)}"
    return None


@app.function(gpu="A100", timeout=12 * 3600, volumes=VOLUMES, secrets=secrets)
def train_queue(jobs: list[dict], wait_for: str = ""):
    if wait_for:
        deadline = time.time() + 4 * 3600
        while time.time() < deadline:
            out_volume.reload()
            if list(Path("/out").glob(wait_for)):
                break
            print(f"[queue] waiting for {wait_for}", flush=True)
            time.sleep(120)
        else:
            return f"gave up waiting for {wait_for}"

    done = []
    for job in jobs:
        stage = job["stage"]
        print(f"[queue] starting {stage}", flush=True)
        code = run_stage(stage, job["extra"].split())
        print(f"[queue] {stage} exit {code}", flush=True)
        if code != 0:
            return f"stopped at {stage} (exit {code}); completed: {done}"
        done.append(stage)
    return f"queue complete: {done}"


@app.function(gpu="A100", timeout=12 * 3600, volumes=VOLUMES, secrets=secrets)
def train_a100(stage: str, extra_args: list[str]):
    return run_stage(stage, extra_args)


@app.function(gpu="H100", timeout=12 * 3600, volumes=VOLUMES, secrets=secrets)
def train_h100(stage: str, extra_args: list[str]):
    return run_stage(stage, extra_args)


EVAL_CHECKPOINTS = [
    ("pretrain_full", 0),
    ("full_sft_full", 0),
    ("dpo_full", 0),
    ("full_dist", 0),
    ("ppo_actor", 0),
    ("grpo", 0),
    ("agent", 0),
    ("full_sft_moe", 1),
]


@app.function(gpu="A100", timeout=6 * 3600, volumes=VOLUMES)
def evaluate(limit: int):
    import json

    env = {**os.environ, "PYTHONPATH": REMOTE_REPO, **CACHE_ENV}
    results = {}
    for weight, use_moe in EVAL_CHECKPOINTS:
        suffix = "_moe" if use_moe else ""
        if not Path(f"/out/{weight}_768{suffix}.pth").exists():
            print(f"[skip] {weight}", flush=True)
            continue
        out_path = f"/out/eval_{weight}.json"
        command = [
            sys.executable, f"{REMOTE_REPO}/scripts/evaluate.py",
            "--weight", weight,
            "--save_dir", "/out",
            "--tokenizer_dir", f"{REMOTE_REPO}/tokenizer",
            "--use_moe", str(use_moe),
            "--device", "cuda:0",
            "--limit", str(limit),
            "--out", out_path,
        ]
        print(f"[eval] {weight}", flush=True)
        if subprocess.run(command, cwd=REMOTE_REPO, env=env).returncode != 0:
            print(f"[eval] {weight} failed", flush=True)
            continue
        with open(out_path) as handle:
            results[weight] = json.load(handle)
        out_volume.commit()
    with open("/out/eval_summary.json", "w") as handle:
        json.dump(results, handle, indent=2)
    out_volume.commit()
    return results


@app.local_entrypoint()
def benchmark(limit: int = 500):
    print(f"evaluation running as {evaluate.spawn(limit).object_id}")


@app.local_entrypoint()
def main(stage: str = "pretrain", gpu: str = "A100", extra: str = "", wait: bool = False):
    extra_args = extra.split()
    from_weight = extra_args[extra_args.index("--from_weight") + 1] if "--from_weight" in extra_args else None
    problem = check_inputs.remote(stage, from_weight)
    if problem:
        raise SystemExit(f"refusing to start {stage} — {problem}")
    train = train_h100 if gpu.upper() == "H100" else train_a100
    if wait:
        print(f"{stage} finished with exit code {train.remote(stage, extra_args)}")
        return
    print(f"{stage} running as {train.spawn(stage, extra_args).object_id}")
