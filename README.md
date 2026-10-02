# Manas

A 64M-parameter language model trained from scratch and carried through every modern training stage: tokenizer,
data pipeline, pretraining, SFT, DPO, PPO, GRPO, agentic RL with tool calls, distillation, and a mixture-of-experts
variant. One codebase, one tokenizer, one dataset repo, the same evals at every stage.

Weights for every stage: [Manas-64M collection](https://huggingface.co/collections/kuluruvineeth/manas-64m-every-training-stage-6abf967eb1940477ad595a8e) ·
data: [`kuluruvineeth/manas_dataset`](https://huggingface.co/datasets/kuluruvineeth/manas_dataset)

## Stages

| Stage | Starts from | Trainer | Held-out loss | 5-benchmark avg |
|---|---|---|---|---|
| pretrain | scratch | `trainer/train_pretrain.py` | 2.145 | 28.6% |
| full_sft | pretrain | `trainer/train_full_sft.py` | 1.375 | 28.0% |
| dpo | full_sft | `trainer/train_dpo.py` | — | 28.0% |
| ppo | full_sft | `trainer/train_ppo.py` (reward: Skywork-Reward-V2-Qwen3-0.6B, KL penalty) | — | 28.0% |
| grpo | full_sft | `trainer/train_grpo.py` (same reward model, group-relative advantages) | — | 28.0% |
| agent | full_sft | `trainer/train_agent.py` (up to 3 tool-calling turns, answer checked against ground truth) | — | 28.0% |
| distillation | full_sft | `trainer/train_distillation.py` (MoE SFT teacher) | 1.131 | 27.2% |
| full_sft_moe | pretrain_moe | `trainer/train_full_sft.py --use_moe 1` | 2.181 | 25.8% |

Benchmarks are ARC-Challenge, ARC-Easy, HellaSwag, MMLU and OpenBookQA, 500 items each. A 64M model sits near
random chance (25%) on all of them; the numbers are published because hiding them would be dishonest. What the model
can do is hold a short conversation, answer simple factual questions and call tools.

## Model

768-wide, 8 layers, 63.9M parameters (the MoE variant is 0.2B total). The architecture matches Qwen3's layer names,
so exported checkpoints are plain `Qwen3ForCausalLM` and load in transformers, vLLM and SGLang with no custom code:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("kuluruvineeth/manas-64m-full-sft")
tokenizer = AutoTokenizer.from_pretrained("kuluruvineeth/manas-64m-full-sft")
```

## Train it yourself

```bash
uv sync
hf download kuluruvineeth/manas_dataset --repo-type dataset --local-dir dataset
uv run python trainer/train_pretrain.py
uv run python trainer/train_full_sft.py
uv run python trainer/train_grpo.py          # or dpo, ppo, agent, distillation
uv run python scripts/evaluate.py
uv run python scripts/convert_model.py --stage full_sft --out_dir manas-64m-hf
```

Each trainer prints its options with `--help`. `gpu/` holds the Modal launchers used for the GPU runs.

## Layout

```text
manas/       model, training loops, data loaders, evals, serving, tools
trainer/     one entry point per stage
datapipe/    builds and publishes the dataset repo
scripts/     evaluate, chat, export, plot metrics, OpenAI-compatible server
deploy/      Docker, Kubernetes and Terraform for serving (see deploy/README.md)
demo/        Chainlit chat UI for any OpenAI-compatible endpoint
tests/       247 tests: `uv run pytest`
```

## License

Apache-2.0
