import os
import subprocess
import sys
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
secrets = [modal.Secret.from_name("wandb-secret")]

STAGE_DATA = {
    "pretrain": "pretrain_t2t_mini.jsonl",
    "pretrain_full": "pretrain_t2t.jsonl",
    "full_sft": "sft_t2t_mini.jsonl",
    "full_sft_full": "sft_t2t.jsonl",
    "lora": "lora_identity.jsonl",
    "dpo": "dpo.jsonl",
    "distillation": "sft_t2t_mini.jsonl",
}


def run_stage(stage, extra_args):
    trainer = stage.removesuffix("_full")
    command = [
        sys.executable, f"{REMOTE_REPO}/trainer/train_{trainer}.py",
        "--data_path", f"/data/{STAGE_DATA[stage]}",
        "--tokenizer_dir", f"{REMOTE_REPO}/tokenizer",
        "--save_dir", "/out",
        "--device", "cuda:0",
        *extra_args,
    ]
    print("running:", " ".join(command), flush=True)
    result = subprocess.run(command, cwd=REMOTE_REPO, env={**os.environ, "PYTHONPATH": REMOTE_REPO})
    out_volume.commit()
    return result.returncode


@app.function(gpu="A100", timeout=8 * 3600, volumes={"/data": data_volume, "/out": out_volume}, secrets=secrets)
def train_a100(stage: str, extra_args: list[str]):
    return run_stage(stage, extra_args)


@app.function(gpu="H100", timeout=8 * 3600, volumes={"/data": data_volume, "/out": out_volume}, secrets=secrets)
def train_h100(stage: str, extra_args: list[str]):
    return run_stage(stage, extra_args)


@app.local_entrypoint()
def main(stage: str = "pretrain", gpu: str = "A100", extra: str = ""):
    train = train_h100 if gpu.upper() == "H100" else train_a100
    code = train.remote(stage, extra.split())
    print(f"{stage} finished with exit code {code}")
