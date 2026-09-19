import argparse

import torch
import uvicorn

from manas.config import ManasConfig
from manas.serve.api import create_app
from manas.training.utils import init_model


def main():
    parser = argparse.ArgumentParser(description="Serve Manas behind an OpenAI-compatible API")
    parser.add_argument("--weight", default="full_sft")
    parser.add_argument("--save_dir", default="out")
    parser.add_argument("--tokenizer_dir", default="tokenizer")
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_hidden_layers", type=int, default=8)
    parser.add_argument("--use_moe", type=int, default=0)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8998)
    args = parser.parse_args()

    config = ManasConfig(
        hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe)
    )
    model, tokenizer = init_model(config, args.weight, args.save_dir, args.device, args.tokenizer_dir)
    model.eval()
    uvicorn.run(create_app(model, tokenizer, args.device), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
