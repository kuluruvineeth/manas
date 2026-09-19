import argparse
import json

import torch

from manas.config import ManasConfig
from manas.eval.benchmarks import BENCHMARKS, load_benchmark
from manas.eval.multiple_choice import accuracy, random_baseline
from manas.training.utils import init_model


def main():
    parser = argparse.ArgumentParser(description="Score a Manas checkpoint on multiple-choice benchmarks")
    parser.add_argument("--weight", default="full_sft")
    parser.add_argument("--save_dir", default="out")
    parser.add_argument("--tokenizer_dir", default="tokenizer")
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_hidden_layers", type=int, default=8)
    parser.add_argument("--use_moe", type=int, default=0)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--benchmarks", nargs="+", default=list(BENCHMARKS))
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    config = ManasConfig(
        hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe)
    )
    model, tokenizer = init_model(config, args.weight, args.save_dir, args.device, args.tokenizer_dir)
    model.eval()

    results = {}
    print(f"{'benchmark':16s} {'accuracy':>9s} {'chance':>7s} {'items':>6s}")
    for name in args.benchmarks:
        records = load_benchmark(name, args.limit)
        score, seen = accuracy(records, model, tokenizer, args.device)
        chance = random_baseline(records)
        results[name] = {"accuracy": score, "chance": chance, "items": seen}
        print(f"{name:16s} {score:>8.1%} {chance:>7.1%} {seen:>6d}")

    average = sum(r["accuracy"] for r in results.values()) / max(len(results), 1)
    print(f"{'average':16s} {average:>8.1%}")
    print("\nSmall models score near chance on knowledge benchmarks. Report the gap, not the headline.")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"weight": args.weight, "results": results, "average": average}, f, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
