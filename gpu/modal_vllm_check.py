import subprocess

import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("vllm==0.11.0", "transformers<5", "huggingface_hub")
)

app = modal.App("manas-vllm-check", image=image)
out_volume = modal.Volume.from_name("manas-out", create_if_missing=True)


@app.function(gpu="A10G", timeout=1800, volumes={"/out": out_volume})
def check():
    subprocess.run(["mkdir", "-p", "/tmp/model"], check=True)
    subprocess.run(["tar", "xzf", "/out/manas-hf.tgz", "-C", "/tmp/model"], check=True)
    print(subprocess.run(["ls", "-la", "/tmp/model"], capture_output=True, text=True).stdout)

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    llm = LLM(
        model="/tmp/model",
        dtype="float16",
        max_model_len=512,
        gpu_memory_utilization=0.6,
        enforce_eager=True,
    )
    tokenizer = AutoTokenizer.from_pretrained("/tmp/model")
    questions = ["What is the capital of France?", "Once upon a time"]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": question}], tokenize=False, add_generation_prompt=True
        )
        for question in questions
    ]
    outputs = llm.generate(prompts, SamplingParams(temperature=0.7, top_p=0.9, max_tokens=48))
    for question, output in zip(questions, outputs, strict=True):
        print("PROMPT:", question)
        print("OUTPUT:", output.outputs[0].text.strip().replace("\n", " ")[:200])
        print("TOKENS:", len(output.outputs[0].token_ids))
    return "vllm loaded and generated"


@app.local_entrypoint()
def main():
    print(check.remote())
