import os

from manas.config import ManasConfig
from manas.data.sft import SFTDataset
from manas.lora import apply_lora, lora_parameters, save_lora
from manas.training.loop import build_parser, fit, language_model_loss
from manas.training.utils import init_model, log, setup_seed


def parse_args(argv=None):
    parser = build_parser(
        "Manas LoRA fine-tuning",
        save_weight="lora_identity",
        batch_size=32,
        learning_rate=1e-4,
        accumulation_steps=1,
        max_seq_len=512,
        data_path="dataset/lora_identity.jsonl",
        from_weight="full_sft",
        lora_rank=16,
    )
    parser.set_defaults(epochs=10, log_interval=10)
    return parser.parse_args(argv)


def save_adapter(args, model):
    os.makedirs(args.save_dir, exist_ok=True)
    save_lora(model, os.path.join(args.save_dir, f"{args.save_weight}_{model.config.hidden_size}.pth"))


def train(args):
    setup_seed(args.seed)
    config = ManasConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers)
    model, tokenizer = init_model(config, args.from_weight, args.save_dir, args.device, args.tokenizer_dir)
    apply_lora(model, rank=args.lora_rank)
    for name, param in model.named_parameters():
        param.requires_grad = ".lora." in name
    trainable = lora_parameters(model)
    log(f"lora rank {args.lora_rank}: {sum(p.numel() for p in trainable):,} trainable parameters")
    dataset = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    return fit(args, model, dataset, language_model_loss, parameters=trainable, save=save_adapter)


if __name__ == "__main__":
    train(parse_args())
