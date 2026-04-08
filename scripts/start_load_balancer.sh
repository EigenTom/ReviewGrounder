#!/bin/bash
# Start Python Load Balancer for vLLM and Reranker services
# Usage: ./scripts/start_load_balancer.sh [service_type] [num_instances] [base_port] [lb_port]

set -e

SERVICE_TYPE="${1:-vllm}"  # vllm or reranker
NUM_INSTANCES="${2:-4}"
BASE_PORT="${3:-8000}"
LB_PORT="${4:-$BASE_PORT}"

echo "Starting Load Balancer for $SERVICE_TYPE"
echo "Number of instances: $NUM_INSTANCES"
echo "Base port: $BASE_PORT"
echo "Load balancer port: $LB_PORT"
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
    echo "Error: FastAPI not installed. Install with: pip install fastapi uvicorn httpx"
    exit 1
}

# Build backend list
BACKENDS=()
for i in $(seq 0 $((NUM_INSTANCES - 1))); do
    PORT=$((BASE_PORT + i))
    if [ "$SERVICE_TYPE" = "vllm" ]; then
        BACKENDS+=("http://localhost:${PORT}/v1")
    else
        BACKENDS+=("http://localhost:${PORT}")
    fi
done

# Create logs directory based on service type
if [ "$SERVICE_TYPE" = "vllm" ]; then
    LB_LOG_DIR="logs/vllm"
else
    LB_LOG_DIR="logs/reranker"
fi
mkdir -p "$LB_LOG_DIR"

echo "Backends:"
for backend in "${BACKENDS[@]}"; do
    echo "  - $backend"
done
echo ""

# Start load balancer
echo "Starting load balancer..."
python3 -m shared.utils.load_balancer \
    --backends "${BACKENDS[@]}" \
    --host 0.0.0.0 \
    --port "$LB_PORT" \
    --strategy round_robin \
    --health-check-interval 10.0 \
    > "${LB_LOG_DIR}/load_balancer_${SERVICE_TYPE}_port${LB_PORT}.log" 2>&1 &

LB_PID=$!

# Save PID to file based on service type
if [ "$SERVICE_TYPE" = "vllm" ]; then
    PID_FILE="logs/vllm/vllm_lb_pid.txt"
    mkdir -p logs/vllm
else
    PID_FILE="logs/reranker/reranker_lb_pid.txt"
    mkdir -p logs/reranker
fi
echo "$LB_PID" > "$PID_FILE"

echo "Load balancer started with PID: $LB_PID"
echo "Load balancer URL: http://localhost:${LB_PORT}"
echo "PID saved to: $PID_FILE"
echo ""
echo "To check status: curl http://localhost:${LB_PORT}/health"
echo "To stop: ./scripts/stop_vllm_services.sh (for vllm) or ./scripts/stop_reranker_services.sh (for reranker)"
echo "Or manually: kill $LB_PID"
