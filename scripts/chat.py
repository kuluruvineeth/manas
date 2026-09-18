import argparse

import torch
from transformers import TextStreamer

from manas.config import ManasConfig
from manas.training.utils import init_model


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Talk to a Manas checkpoint")
    parser.add_argument("--weight", default="full_sft")
    parser.add_argument("--save_dir", default="out")
    parser.add_argument("--tokenizer_dir", default="tokenizer")
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_hidden_layers", type=int, default=8)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--open_thinking", type=int, default=0)
    parser.add_argument("--history", type=int, default=0)
    parser.add_argument("--prompt", default=None)
    return parser.parse_args(argv)


def build_input(tokenizer, weight, history, open_thinking):
    if "pretrain" in weight:
        return tokenizer.bos_token + history[-1]["content"]
    return tokenizer.apply_chat_template(
        history, tokenize=False, add_generation_prompt=True, open_thinking=bool(open_thinking)
    )


def reply(model, tokenizer, args, history, stream=True):
    text = build_input(tokenizer, args.weight, history, args.open_thinking)
    ids = tokenizer(text, return_tensors="pt").input_ids.to(args.device)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True) if stream else None
    out = model.generate(
        ids, max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_p=args.top_p, streamer=streamer
    )
    return tokenizer.decode(out[0][ids.shape[1] :], skip_special_tokens=True).strip()


def main():
    args = parse_args()
    config = ManasConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers)
    model, tokenizer = init_model(config, args.weight, args.save_dir, args.device, args.tokenizer_dir)
    model.eval()
    history = []
    prompts = [args.prompt] if args.prompt else iter(lambda: input("you: "), "")
    for prompt in prompts:
        history.append({"role": "user", "content": prompt})
        print("manas: ", end="", flush=True)
        answer = reply(model, tokenizer, args, history)
        print()
        history.append({"role": "assistant", "content": answer})
        history = history[-args.history * 2 :] if args.history else []


if __name__ == "__main__":
    main()
