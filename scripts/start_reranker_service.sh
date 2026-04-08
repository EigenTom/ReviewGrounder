#!/bin/bash
# Start Reranker API Service on multiple GPUs
# Usage: ./scripts/start_reranker_service.sh [model_path] [num_gpus] [base_port]

set -e

# Default values
# MODEL_PATH="${1:-OpenScholar/OpenScholar_Reranker}"
# NUM_GPUS="${2:-4}"
# BASE_PORT="${3:-8005}"

# MODEL_PATH="BAAI/bge-reranker-base"
# MODEL_PATH="BAAI/bge-reranker-large"
MODEL_PATH="${1:-OpenScholar/OpenScholar_Reranker}"
NUM_GPUS=8
BASE_PORT=8008

echo "Starting Reranker API Service"
echo "Model: $MODEL_PATH"
echo "Number of GPUs: $NUM_GPUS"
echo "Base port: $BASE_PORT"
echo ""

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Check if FastAPI is installed
python3 -c "import fastapi" 2>/dev/null || {
    echo "Error: FastAPI not installed. Install with: pip install fastapi uvicorn"
    exit 1
}

# Check if FlagEmbedding is installed
python3 -c "from FlagEmbedding import FlagReranker" 2>/dev/null || {
    echo "Error: FlagEmbedding not installed. Install with: pip install FlagEmbedding"
    exit 1
}

# Create logs directory
mkdir -p logs

# PID file for stopping services later
PID_FILE="logs/reranker/reranker_pids.txt"
LB_PID_FILE="logs/reranker/reranker_lb_pid.txt"

# Start services on each GPU
PIDS=()
ENDPOINTS=()

# Ensure we use GPUs 0, 1, 2, 3 (explicitly)

# now we use gpus: 1,2,3,4,5,6,7
for i in $(seq 0 $((NUM_GPUS - 1))); do
    PORT=$((BASE_PORT + i))
    GPU_ID=$i  # Use GPU 0, 1, 2, 3 explicitly
    
    echo "Starting reranker service on GPU $GPU_ID, port $PORT..."
    
    # Set CUDA device (each service will see only one GPU)
    export CUDA_VISIBLE_DEVICES=$GPU_ID
    
    # Start service in background
    # Note: When CUDA_VISIBLE_DEVICES is set, cuda:0 refers to the visible GPU
    nohup python3 -m shared.utils.reranker_api_service \
        --model_path "$MODEL_PATH" \
        --host 0.0.0.0 \
        --port "$PORT" \
        --use_fp16 \
        --device "cuda:0" \
        > "logs/reranker/reranker_service_gpu${GPU_ID}_port${PORT}.log" 2>&1 &
    
    PID=$!
    PIDS+=($PID)
    ENDPOINTS+=("http://localhost:${PORT}")
    
    echo "  Started with PID: $PID"
    echo "  Endpoint: http://localhost:${PORT}"
    sleep 2  # Give service time to start
done

echo ""
echo "All reranker services started!"
echo ""
echo "Endpoints:"
for endpoint in "${ENDPOINTS[@]}"; do
    echo "  - $endpoint"
done

# Create endpoint pool file
ENDPOINT_POOL_FILE="shared/configs/reranker_endpoint_pool.txt"
mkdir -p "$(dirname "$ENDPOINT_POOL_FILE")"
printf "%s\n" "${ENDPOINTS[@]}" > "$ENDPOINT_POOL_FILE"
echo ""
echo "Endpoint pool file created: $ENDPOINT_POOL_FILE"

# Save PIDs to file (one per line)
printf "%s\n" "${PIDS[@]}" > "$PID_FILE"
echo ""
echo "PIDs saved to: $PID_FILE"
echo ""
echo "To stop these specific reranker services, run:"
echo "  ./scripts/stop_reranker_services.sh"
echo ""
echo "This will only kill the processes listed above, not other reranker services."
echo ""
echo "To check service status, run:"
for endpoint in "${ENDPOINTS[@]}"; do
    echo "curl $endpoint/health"
done
