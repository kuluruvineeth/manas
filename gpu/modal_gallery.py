from pathlib import Path

import modal

REPO = Path(__file__).resolve().parents[1]
REMOTE_REPO = "/root/manas"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("torch", "transformers", "tokenizers", "jinja2", "numpy", "fastapi[standard]")
    .add_local_dir(
        str(REPO),
        remote_path=REMOTE_REPO,
        copy=True,
        ignore=[".venv", ".git", "dataset", "out", "checkpoints", "__pycache__", ".ruff_cache", ".pytest_cache"],
    )
)

app = modal.App("manas-gallery", image=image)
out_volume = modal.Volume.from_name("manas-out", create_if_missing=True)

GALLERY = [
    ("pretrain", "pretrain_full", 0),
    ("sft", "full_sft_full", 0),
    ("dpo", "dpo_full", 0),
    ("distilled", "full_dist", 0),
    ("ppo", "ppo_actor", 0),
    ("grpo", "grpo", 0),
    ("agent", "agent", 0),
    ("moe", "full_sft_moe", 1),
    ("sft-mini", "full_sft", 0),
    ("pretrain-mini", "pretrain", 0),
]

ADAPTERS = [("identity", "full_sft_full", "lora_identity_mixed")]


@app.function(gpu="A10G", timeout=3600, volumes={"/out": out_volume}, scaledown_window=900)
@modal.concurrent(max_inputs=8)
@modal.asgi_app()
def gallery():
    import sys

    sys.path.insert(0, REMOTE_REPO)

    from manas.config import ManasConfig
    from manas.lora import apply_lora, load_lora
    from manas.serve.api import create_gallery_app
    from manas.training.utils import init_model, weight_path

    models, tokenizer = {}, None
    for label, weight, use_moe in GALLERY:
        config = ManasConfig(use_moe=bool(use_moe))
        try:
            model, tokenizer = init_model(config, weight, "/out", "cuda:0", f"{REMOTE_REPO}/tokenizer")
        except FileNotFoundError:
            print(f"[gallery] skipping {label}: no checkpoint", flush=True)
            continue
        models[label] = model.eval()
        print(f"[gallery] loaded {label} from {weight}", flush=True)

    for label, base, adapter in ADAPTERS:
        config = ManasConfig()
        try:
            model, tokenizer = init_model(config, base, "/out", "cuda:0", f"{REMOTE_REPO}/tokenizer")
            apply_lora(model, rank=16)
            load_lora(model, weight_path("/out", adapter, config))
        except FileNotFoundError:
            print(f"[gallery] skipping {label}: no adapter", flush=True)
            continue
        models[label] = model.eval()
        print(f"[gallery] loaded {label} = {base} + {adapter}", flush=True)

    print(f"[gallery] serving {sorted(models)}", flush=True)
    return create_gallery_app(models, tokenizer, "cuda:0")
