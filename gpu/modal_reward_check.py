import modal

image = modal.Image.debian_slim(python_version="3.12").uv_pip_install(
    "torch", "transformers", "huggingface_hub", "hf_transfer"
)

app = modal.App("manas-reward-check", image=image)
cache_volume = modal.Volume.from_name("manas-hf-cache", create_if_missing=True)

MODEL = "Skywork/Skywork-Reward-V2-Qwen3-0.6B"
QUESTION = "What is the capital of France?"
ANSWERS = [
    "Paris is the capital of France.",
    "The capital of France is France.",
    "Bananas are yellow.",
]


@app.function(gpu="A10G", timeout=1800, volumes={"/cache": cache_volume}, env={"HF_HOME": "/cache/huggingface"})
def check():
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = (
        AutoModelForSequenceClassification.from_pretrained(MODEL, dtype=torch.float16, num_labels=1)
        .to("cuda:0")
        .eval()
    )

    scores = []
    for answer in ANSWERS:
        conversation = [{"role": "user", "content": QUESTION}, {"role": "assistant", "content": answer}]
        text = tokenizer.apply_chat_template(conversation, tokenize=False)
        tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096).to("cuda:0")
        with torch.no_grad():
            score = model(**tokens).logits[0][0].item()
        scores.append(score)
        print(f"SCORE {score:+.2f}  {answer}")

    cache_volume.commit()
    ordered = scores == sorted(scores, reverse=True)
    print(f"ORDERING CORRECT: {ordered}")
    return "reward model loaded and ranked correctly" if ordered else f"WRONG ORDER: {scores}"


@app.local_entrypoint()
def main():
    print(check.remote())
