# Serving Manas

Three ways to run the model, in increasing order of effort and throughput.

| | engine | when to use it |
|---|---|---|
| 1 | **our own FastAPI server** (`scripts/serve_openai_api.py`) | development, debugging, understanding every line |
| 2 | **vLLM** | production serving of one model, continuous batching, the default choice |
| 3 | **SGLang** | high-throughput RL rollouts and structured generation |

TensorRT-LLM is the fourth option and is deliberately not used here: it compiles a model into a
hardware-specific engine, which buys latency on NVIDIA hardware at the cost of a build step and
portability. For a 64M model the wins are irrelevant.

Our exported checkpoint is a genuine `Qwen3ForCausalLM`, so every engine below loads it with no
custom code — that is the whole payoff of matching Qwen3's layer names in the architecture.

## 1. Export first

```bash
uv run python scripts/convert_model.py --stage full_sft --out_dir manas-64m-hf
```

## 2. vLLM

```bash
pip install vllm
vllm serve manas-64m-hf --served-model-name manas-64m --max-model-len 2048 --port 8000
```

Then talk to it with any OpenAI client:

```bash
curl localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "manas-64m",
  "messages": [{"role": "user", "content": "What is the capital of France?"}]
}'
```

Useful flags: `--gpu-memory-utilization 0.9`, `--max-num-seqs 256` (concurrency),
`--tensor-parallel-size N` (split across N GPUs), `--enable-prefix-caching` (shared system prompts).

## 3. SGLang

```bash
pip install "sglang[all]"
python -m sglang.launch_server --model-path manas-64m-hf --port 30000
```

SGLang is also what the RL rollout path would use at scale: generation runs in a separate server,
weights are pushed to it between updates instead of generating inside the training process.

## 4. Docker

```bash
docker build -f deploy/Dockerfile -t manas-serve:latest .
docker run --gpus all -p 8000:8000 manas-serve:latest
```

The image contains only the exported weights and vLLM — no training code, no datasets.

## 5. Kubernetes

The manifests need a cluster with GPU nodes. `deploy/terraform/` builds one; start there.

```bash
# the adapter that lets an HPA read a Prometheus metric on GKE
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/k8s-stackdriver/master/custom-metrics-stackdriver-adapter/deploy/production/adapter_new_resource_model.yaml

kubectl apply -f deploy/k8s/
kubectl port-forward svc/manas 8000:80
```

What the manifests actually do, and why each piece exists:

- **`nvidia.com/gpu: 1`** — the node must expose GPUs, or the pod stays `Pending` forever with no
  obvious error. This is the single most common first failure. On GKE the device plugin and the
  driver are managed for you; elsewhere you install the NVIDIA device plugin or GPU Operator.
- **startup probe on `/health`** — loading weights takes tens of seconds. Without it the liveness
  probe kills the container mid-load, forever.
- **readiness probe** — keeps traffic away from a pod that cannot answer yet.
- **liveness probe** — restarts a pod whose CUDA context has died but whose process is alive.
- **`emptyDir` on `/dev/shm`** — tensor-parallel serving communicates over shared memory, and
  Kubernetes' 64MB default is too small. Silent hangs come from this.
- **resource requests == limits** — GPUs cannot be oversubscribed; asking for a fraction is invalid.
- **`strategy: Recreate`** — a rolling update wants a surge pod, which needs a *spare* GPU. On a
  cluster whose GPUs are all busy that pod stays `Pending` and the rollout never finishes.
- **`PodMonitoring`** — without something scraping `/metrics`, the autoscaler's metric does not
  exist and the HPA sits at `<unknown>` forever.

### Autoscale on queue depth, not GPU utilisation

The obvious-looking choice is wrong, and it is worth understanding why. `DCGM_FI_DEV_GPU_UTIL`
measures the fraction of time at least one kernel was resident — occupancy, not work done. A
decode loop with a single sequence pins it near 100% while the GPU is memory-bandwidth bound and
arithmetically almost idle. It saturates long before the server does, so it reports no headroom
when plenty remains, and scales up capacity that buys nothing. Google's guidance is blunt: *"we do
not recommend using GPU utilization for autoscaling inference workloads."*

`vllm:num_requests_waiting` is the honest signal — requests are queued only when demand genuinely
exceeds capacity. Start the target between 3 and 5 and raise it until latency reaches the highest
you will accept. Its weakness is that it lags: it sits at zero right up to the cliff. If you want
a leading signal, `vllm:kv_cache_usage_perc` fills up first and causes the queueing (note it is a
0–1 fraction, not a percentage).

Two traps. Google's own vLLM tutorial still references `gpu_cache_usage_perc`, which no longer
exists — it was renamed to `kv_cache_usage_perc`, so an HPA copied from that page silently finds
no metric. And every vLLM metric carries `model_name` and `engine` labels, so a query must `sum()`
or it returns a vector rather than a scalar.

Off GKE, KEDA is the more common path: it queries Prometheus directly and needs no metrics adapter,
and it supports scaling to zero. The tradeoff there is paying a full model load on the next request.

## 6. What actually matters in production

- **Continuous batching** is the reason to use vLLM at all: requests join and leave the batch mid-flight
  instead of waiting for the slowest one.
- **Paged attention** stores the KV cache in fixed-size blocks so memory is not fragmented by variable
  sequence lengths — this is what lets a server hold many concurrent conversations.
- **Prefix caching** reuses the KV cache for a shared system prompt across requests.
- **Quantisation** (AWQ, GPTQ, FP8) trades a little quality for memory and speed; irrelevant for 64M,
  essential at 7B+.
- **Measure**: time to first token, inter-token latency, and tokens/sec at your real concurrency —
  not single-request speed, which flatters every engine equally.
