import argparse
import json
import os

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM, Qwen3MoeConfig, Qwen3MoeForCausalLM

from manas.config import ManasConfig
from manas.training.utils import init_model, weight_path


def qwen_config(config):
    shared = dict(
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        max_position_embeddings=config.max_position_embeddings,
        rms_norm_eps=config.rms_norm_eps,
        rope_theta=config.rope_theta,
        hidden_act=config.hidden_act,
        tie_word_embeddings=config.tie_word_embeddings,
        bos_token_id=config.bos_token_id,
        eos_token_id=config.eos_token_id,
    )
    if not config.use_moe:
        return Qwen3Config(intermediate_size=config.intermediate_size, **shared)
    return Qwen3MoeConfig(
        intermediate_size=config.intermediate_size,
        moe_intermediate_size=config.moe_intermediate_size,
        num_experts=config.num_experts,
        num_experts_per_tok=config.num_experts_per_tok,
        norm_topk_prob=config.norm_topk_prob,
        decoder_sparse_step=1,
        **shared,
    )


def fuse_experts(state, config, target_state):
    fused = {k: v for k, v in state.items() if ".mlp.experts." not in k}
    for layer in range(config.num_hidden_layers):
        prefix = f"model.layers.{layer}.mlp.experts"
        stacks = {"gate_up_proj": [], "down_proj": []}
        for expert in range(config.num_experts):
            gate = state[f"{prefix}.{expert}.gate_proj.weight"]
            up = state[f"{prefix}.{expert}.up_proj.weight"]
            down = state[f"{prefix}.{expert}.down_proj.weight"]
            stacks["gate_up_proj"].append(torch.cat([gate, up], dim=0).t())
            stacks["down_proj"].append(down.t())
        for name, tensors in stacks.items():
            stacked = torch.stack(tensors)
            expected = target_state[f"{prefix}.{name}"].shape
            if stacked.shape != expected:
                stacked = stacked.transpose(1, 2)
            fused[f"{prefix}.{name}"] = stacked.contiguous()
    return fused


V5_ONLY_KEYS = (
    "extra_special_tokens",
    "backend",
    "is_local",
    "local_files_only",
    "model_specific_special_tokens",
    "audio_bos_token",
    "audio_eos_token",
    "audio_token",
    "image_token",
    "video_token",
    "vision_bos_token",
    "vision_eos_token",
)


EMPTY_THINK_BRANCH = """    {%- else %}
        {{- '<think>\\n\\n</think>\\n\\n' }}
    {%- endif %}"""
NO_EMPTY_THINK_BRANCH = """    {%- endif %}"""


def training_shaped_chat_template(out_dir):
    path = os.path.join(out_dir, "chat_template.jinja")
    with open(path) as handle:
        template = handle.read()
    if EMPTY_THINK_BRANCH not in template:
        raise SystemExit("chat template no longer has the empty-think branch to remove")
    with open(path, "w") as handle:
        handle.write(template.replace(EMPTY_THINK_BRANCH, NO_EMPTY_THINK_BRANCH))


def portable_tokenizer_config(out_dir):
    path = os.path.join(out_dir, "tokenizer_config.json")
    with open(path) as handle:
        config = json.load(handle)
    for key in V5_ONLY_KEYS:
        config.pop(key, None)
    config["tokenizer_class"] = "PreTrainedTokenizerFast"
    with open(path, "w") as handle:
        json.dump(config, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def export(stage, config, save_dir, tokenizer_dir, out_dir, device="cpu"):
    model, tokenizer = init_model(config, stage, save_dir, device, tokenizer_dir)
    target_config = qwen_config(config)
    target = Qwen3MoeForCausalLM(target_config) if config.use_moe else Qwen3ForCausalLM(target_config)
    state = {k: v for k, v in model.state_dict().items() if "freqs" not in k}
    if config.use_moe:
        state = fuse_experts(state, config, target.state_dict())
    missing, unexpected = target.load_state_dict(state, strict=False)
    assert not unexpected, f"unexpected keys: {unexpected[:5]}"
    assert all("rotary" in k or "inv_freq" in k for k in missing), f"missing keys: {missing[:5]}"
    os.makedirs(out_dir, exist_ok=True)
    target.half().save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    portable_tokenizer_config(out_dir)
    training_shaped_chat_template(out_dir)
    return out_dir


def main():
    parser = argparse.ArgumentParser(description="Export a Manas checkpoint to the Qwen3 format")
    parser.add_argument("--stage", default="full_sft")
    parser.add_argument("--save_dir", default="out")
    parser.add_argument("--tokenizer_dir", default="tokenizer")
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_hidden_layers", type=int, default=8)
    parser.add_argument("--use_moe", type=int, default=0)
    args = parser.parse_args()
    config = ManasConfig(
        hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe)
    )
    out_dir = args.out_dir or f"manas-64m-{args.stage.replace('_', '-')}-hf"
    export(args.stage, config, args.save_dir, args.tokenizer_dir, out_dir)
    print(f"[done] {out_dir}")
    print(f"       weights from {weight_path(args.save_dir, args.stage, config)}")
    print(json.dumps({"architectures": ["Qwen3MoeForCausalLM" if args.use_moe else "Qwen3ForCausalLM"]}))


if __name__ == "__main__":
    main()
