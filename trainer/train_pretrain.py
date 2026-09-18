from manas.config import ManasConfig
from manas.data.pretrain import PretrainDataset
from manas.training.loop import build_parser, fit, language_model_loss
from manas.training.utils import init_model, setup_seed


def parse_args(argv=None):
    parser = build_parser(
        "Manas pretraining",
        save_weight="pretrain",
        batch_size=32,
        learning_rate=5e-4,
        accumulation_steps=8,
        max_seq_len=512,
        data_path="dataset/pretrain_t2t_mini.jsonl",
        from_weight="none",
    )
    return parser.parse_args(argv)


def train(args):
    setup_seed(args.seed)
    config = ManasConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers)
    model, tokenizer = init_model(config, args.from_weight, args.save_dir, args.device, args.tokenizer_dir)
    dataset = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    return fit(args, model, dataset, language_model_loss)


if __name__ == "__main__":
    train(parse_args())
