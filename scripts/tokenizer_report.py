import argparse

from transformers import AutoTokenizer

SAMPLES = [
    "The quick brown fox jumps over the lazy dog.",
    "Manas is a small language model, trained from scratch on English text.",
    "def add(a, b):\n    return a + b",
    "In 1969, astronauts landed on the Moon after a journey of about 384,400 kilometres.",
    "<|im_start|>user\nWhat is 23 * 7?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n161<|im_end|>\n",
]


def main():
    parser = argparse.ArgumentParser(description="Show how the Manas tokenizer splits text")
    parser.add_argument("--tokenizer", default="tokenizer")
    parser.add_argument("text", nargs="?")
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    ids = (tokenizer.pad_token_id, tokenizer.bos_token_id, tokenizer.eos_token_id)
    print(f"vocab {len(tokenizer):,} · pad {ids[0]} · bos {ids[1]} · eos {ids[2]}\n")
    for text in [args.text] if args.text else SAMPLES:
        ids = tokenizer.encode(text)
        pieces = [tokenizer.decode([i]) for i in ids]
        print(f"{len(text):4d} chars → {len(ids):3d} tokens  ({len(text) / len(ids):.1f} chars/token)")
        print("  " + " | ".join(p.replace("\n", "⏎") for p in pieces))
        print()


if __name__ == "__main__":
    main()
