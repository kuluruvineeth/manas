import argparse
import time

from manas.tokenizer import VOCAB_SIZE, iter_corpus, save_tokenizer, train_tokenizer


def main():
    parser = argparse.ArgumentParser(description="Train the Manas byte-level BPE tokenizer")
    parser.add_argument("--data", nargs="+", default=["dataset/pretrain_t2t_mini.jsonl", "dataset/sft_t2t_mini.jsonl"])
    parser.add_argument("--out", default="tokenizer")
    parser.add_argument("--vocab-size", type=int, default=VOCAB_SIZE)
    parser.add_argument("--max-docs-per-file", type=int, default=200000)
    args = parser.parse_args()

    started = time.time()
    tokenizer = train_tokenizer(iter_corpus(args.data, args.max_docs_per_file), args.vocab_size)
    path = save_tokenizer(tokenizer, args.out)
    print(f"vocab {tokenizer.get_vocab_size():,} · saved {path} · {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
