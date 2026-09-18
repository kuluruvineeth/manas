from functools import partial

from manas.data.dpo import DPODataset
from manas.training.dpo import preference_loss
from manas.training.loop import build_parser, fit, model_config
from manas.training.utils import init_model, setup_seed


def parse_args(argv=None):
    parser = build_parser(
        "Manas direct preference optimization",
        save_weight="dpo",
        batch_size=4,
        learning_rate=4e-8,
        accumulation_steps=1,
        max_seq_len=1024,
        data_path="dataset/dpo.jsonl",
        from_weight="full_sft",
        beta=0.15,
    )
    parser.set_defaults(epochs=1, save_interval=100)
    return parser.parse_args(argv)


def train(args):
    setup_seed(args.seed)
    config = model_config(args)
    model, tokenizer = init_model(config, args.from_weight, args.save_dir, args.device, args.tokenizer_dir)
    ref_model, _ = init_model(config, args.from_weight, args.save_dir, args.device, args.tokenizer_dir)
    ref_model.eval().requires_grad_(False)
    dataset = DPODataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    return fit(args, model, dataset, partial(preference_loss, ref_model=ref_model, beta=args.beta))


if __name__ == "__main__":
    train(parse_args())
