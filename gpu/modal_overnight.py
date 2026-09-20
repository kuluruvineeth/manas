from modal_train import app, train_queue  # noqa: F401  (app must be imported for `modal run`)

SFT = "--from_weight full_sft_full"

MOE_CHAIN = [
    {
        "stage": "pretrain",
        "extra": "--use_moe 1 --epochs 1 --save_weight pretrain_moe --log_interval 500 "
        "--eval_interval 5000 --save_interval 5000 --val_samples 512 --num_workers 8 "
        "--use_wandb 1 --run_name pretrain-moe",
    },
    {
        "stage": "full_sft",
        "extra": "--use_moe 1 --from_weight pretrain_moe --save_weight full_sft_moe --max_steps 10000 "
        "--log_interval 500 --eval_interval 2000 --save_interval 2000 --val_samples 512 "
        "--num_workers 8 --use_wandb 1 --run_name sft-moe",
    },
    {
        "stage": "distillation",
        "extra": f"{SFT} --teacher_weight full_sft_moe --teacher_use_moe 1 --save_weight full_dist "
        f"--max_steps 3000 --log_interval 200 --eval_interval 1000 --save_interval 1000 "
        f"--val_samples 512 --num_workers 8 --use_wandb 1 --run_name distill-full-tier",
    },
]

RL_CHAIN = [
    {
        "stage": "ppo",
        "extra": f"{SFT} --save_weight ppo_actor --max_steps 900 --log_interval 50 "
        f"--save_interval 300 --use_wandb 1 --run_name ppo-full-tier",
    },
    {
        "stage": "grpo",
        "extra": f"{SFT} --save_weight grpo --max_steps 600 --log_interval 50 "
        f"--save_interval 200 --use_wandb 1 --run_name grpo-full-tier",
    },
    {
        "stage": "agent",
        "extra": f"{SFT} --save_weight agent --max_steps 800 --log_interval 50 "
        f"--save_interval 300 --use_wandb 1 --run_name agent-full-tier",
    },
]


@app.local_entrypoint()
def launch():
    rl = train_queue.spawn(RL_CHAIN)
    print(f"RL chain (ppo -> grpo -> agent) running as {rl.object_id}")
    moe = train_queue.spawn(MOE_CHAIN)
    print(f"MoE chain (pretrain-moe -> sft-moe -> distillation) running as {moe.object_id}")
