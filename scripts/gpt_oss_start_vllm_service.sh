# Script to start vLLM service for Qwen3-235B-A22B-Instruct-2507

# optional: limit GPU usage
# export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_VISIBLE_DEVICES=4,5,6,7

# Configuration
# MODEL_NAME="Qwen/Qwen3-235B-A22B-Instruct-2507"
MODEL_NAME="openai/gpt-oss-120b"
PORT=${VLLM_PORT:-8000}
TP_SIZE=${TP_SIZE:-4}  # Tensor parallelism size, smaller or equal to the number of available GPUs
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.85} # ideally 0.85
MAX_MODEL_LEN=${MAX_MODEL_LEN:-131072}  # Native context length, can extend to 1010000

# Check if model path is provided
if [ -z "$MODEL_PATH" ]; then
    MODEL_PATH="$MODEL_NAME"
    echo "Using HuggingFace model: $MODEL_PATH"
else
    echo "Using local model: $MODEL_PATH"
fi

echo "Starting vLLM service..."
echo "Model: $MODEL_PATH"
echo "Port: $PORT"
echo "Tensor Parallelism: $TP_SIZE"
echo "GPU Memory Utilization: $GPU_MEMORY_UTILIZATION"
echo "Max Model Length: $MAX_MODEL_LEN"

# python3 -m vllm.entrypoints.openai.api_server \
#     --model "$MODEL_PATH" \
#     --port $PORT \
#     --tensor-parallel-size $TP_SIZE \
#     --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
#     --max-model-len $MAX_MODEL_LEN \
#     --trust-remote-code \
#     # --dtype bfloat16


vllm serve openai/gpt-oss-120b \
    --port $PORT \
    --tensor-parallel-size $TP_SIZE \
    --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
    --max-model-len $MAX_MODEL_LEN \
    --trust-remote-code \
    --dtype bfloat16

