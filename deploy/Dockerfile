FROM vllm/vllm-openai:v0.11.0

ARG MODEL_DIR=manas-64m-hf
COPY ${MODEL_DIR} /models/manas

ENV MODEL_PATH=/models/manas \
    SERVED_NAME=manas-64m \
    MAX_MODEL_LEN=2048 \
    GPU_MEMORY_UTILIZATION=0.85

EXPOSE 8000

ENTRYPOINT ["/bin/sh", "-c", "exec python3 -m vllm.entrypoints.openai.api_server \
  --model $MODEL_PATH \
  --served-model-name $SERVED_NAME \
  --max-model-len $MAX_MODEL_LEN \
  --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
  --host 0.0.0.0 --port 8000"]
