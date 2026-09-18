from functools import partial

from manas.config import ManasConfig
from manas.data.sft import SFTDataset
from manas.training.distill import distill_step
from manas.training.loop import build_parser, fit, model_config
from manas.training.utils import init_model, setup_seed


def parse_args(argv=None):
    parser = build_parser(
        "Manas knowledge distillation",
        save_weight="full_dist",
        batch_size=32,
        learning_rate=5e-6,
        accumulation_steps=1,
        max_seq_len=512,
        data_path="dataset/sft_t2t_mini.jsonl",
        from_weight="full_sft",
        teacher_weight="full_sft",
        teacher_hidden_size=768,
        teacher_num_layers=8,
        teacher_use_moe=1,
        alpha=0.5,
        temperature=1.5,
    )
    parser.set_defaults(epochs=6, save_interval=100)
    return parser.parse_args(argv)


def train(args):
    setup_seed(args.seed)
    config = model_config(args)
    student, tokenizer = init_model(config, args.from_weight, args.save_dir, args.device, args.tokenizer_dir)
    teacher_config = ManasConfig(
        hidden_size=args.teacher_hidden_size,
        num_hidden_layers=args.teacher_num_layers,
        use_moe=bool(args.teacher_use_moe),
    )
    teacher, _ = init_model(teacher_config, args.teacher_weight, args.save_dir, args.device, args.tokenizer_dir)
    teacher.eval().requires_grad_(False)
    dataset = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    compute_loss = partial(distill_step, teacher=teacher, alpha=args.alpha, temperature=args.temperature)
    return fit(args, student, dataset, compute_loss)


if __name__ == "__main__":
    train(parse_args())
