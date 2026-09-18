from manas.data.sft import SFTDataset
from manas.training.loop import build_parser, fit, language_model_loss, model_config
from manas.training.utils import init_model, setup_seed


def parse_args(argv=None):
    parser = build_parser(
        "Manas supervised fine-tuning",
        save_weight="full_sft",
        batch_size=16,
        learning_rate=1e-5,
        accumulation_steps=1,
        max_seq_len=768,
        data_path="dataset/sft_t2t_mini.jsonl",
        from_weight="pretrain",
    )
    return parser.parse_args(argv)


def train(args):
    setup_seed(args.seed)
    config = model_config(args)
    model, tokenizer = init_model(config, args.from_weight, args.save_dir, args.device, args.tokenizer_dir)
    dataset = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    return fit(args, model, dataset, language_model_loss)


if __name__ == "__main__":
    train(parse_args())
