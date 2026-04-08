#!/bin/bash
# Script to start vLLM services on GPU 4,5,6,7 and load balancer
# Usage: ./scripts/start_vllm_with_balancer.sh

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Configuration
# MODEL_NAME="ZhuofengLi/Qwen3-4B-Instruct-2507-DeepReview-lora-sft" #
MODEL_NAME="openai/gpt-oss-120b"
# GPU_CONFIG="2:8001,3:8002,4:8003,5:8004"  # GPU:PORT pairs
# GPU_CONFIG="1:8001,2:8002,3:8003,4:8004, 5:8005, 6:8006, 7:8007, 0:7999"  # GPU:PORT pairs
GPU_CONFIG="0:7001,1:7002,2:7003,3:7004,4:7005,5:7006,6:7007,7:7008"  # GPU:PORT pairs
TP_SIZE=1  # Tensor parallelism size per instance
GPU_MEMORY_UTILIZATION=0.9
MAX_MODEL_LEN=131072

# Load balancer configuration
LB_PORT=7000  # Load balancer port
LB_STRATEGY="round_robin"  # or "least_conn"
LB_HEALTH_CHECK_INTERVAL=10.0

# Log directory
LOG_DIR="./logs/vllm"
mkdir -p "$LOG_DIR"

# Endpoint pool file
ENDPOINT_POOL_FILE="shared/configs/vllm_endpoint_pool.txt"
mkdir -p "$(dirname "$ENDPOINT_POOL_FILE")"

echo "=========================================="
echo "Starting vLLM Services + Load Balancer"
echo "=========================================="
echo "Model: $MODEL_NAME"
echo "GPU Configuration: $GPU_CONFIG"
echo "Load Balancer Port: $LB_PORT"
echo "Log Directory: $LOG_DIR"
echo ""

# Step 1: Start vLLM services
echo "=== Step 1: Starting vLLM services ==="
echo ""

# Clear existing endpoints
> "$ENDPOINT_POOL_FILE"

# Parse GPU configuration
IFS=',' read -ra GPU_CONFIGS <<< "$GPU_CONFIG"

# Array to store PIDs
VLLM_PIDS=()

for gpu_config in "${GPU_CONFIGS[@]}"; do
    IFS=':' read -r gpu_id port <<< "$gpu_config"
    
    echo "Starting vLLM on GPU $gpu_id, port $port..."
    
    # Set CUDA_VISIBLE_DEVICES for this specific GPU
    export CUDA_VISIBLE_DEVICES=$gpu_id
    
    # Log file
    LOG_FILE="$LOG_DIR/vllm_gpu${gpu_id}_port${port}.log"
    
    # Start vLLM service in background
    (
        echo "=== GPU $gpu_id, Port $port ===" >> "$LOG_FILE"
        echo "Starting at $(date)" >> "$LOG_FILE"
        echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" >> "$LOG_FILE"
        echo "" >> "$LOG_FILE"
        
        vllm serve "$MODEL_NAME" \
            --port "$port" \
            --tensor-parallel-size "$TP_SIZE" \
            --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
            --max-model-len "$MAX_MODEL_LEN" \
            --trust-remote-code \
            --dtype bfloat16 \
            >> "$LOG_FILE" 2>&1
    ) &
    
    PID=$!
    VLLM_PIDS+=($PID)
    
    # Add endpoint to pool file (for load balancer)
    echo "http://localhost:$port/v1" >> "$ENDPOINT_POOL_FILE"
    
    echo "  -> Started with PID $PID"
    echo "  -> Endpoint: http://localhost:$port/v1"
    echo "  -> Log: $LOG_FILE"
    
    # Wait a bit before starting next service
    sleep 3
done

# Save PIDs (one per line for easier parsing)
printf "%s\n" "${VLLM_PIDS[@]}" > "$LOG_DIR/vllm_pids.txt"
echo ""
echo "vLLM service PIDs saved to: $LOG_DIR/vllm_pids.txt"
echo ""

# Step 2: Wait for services to be ready
echo "=== Step 2: Waiting for vLLM services to be ready ==="
echo "Waiting 90 seconds for services to initialize..."
sleep 90

# Check service health
echo ""
echo "Checking service health..."
HEALTHY_COUNT=0
for gpu_config in "${GPU_CONFIGS[@]}"; do
    IFS=':' read -r gpu_id port <<< "$gpu_config"
    if curl -s "http://localhost:$port/v1/models" > /dev/null 2>&1; then
        echo "  GPU $gpu_id (port $port): HEALTHY"
        HEALTHY_COUNT=$((HEALTHY_COUNT + 1))
    else
        echo "  GPU $gpu_id (port $port): NOT READY (may still be initializing)"
    fi
done

if [ $HEALTHY_COUNT -eq 0 ]; then
    echo ""
    echo "WARNING: No services are healthy yet. They may still be loading the model."
    echo "You can check logs in $LOG_DIR/ for progress."
fi

echo ""

# Step 3: Start load balancer
echo "=== Step 3: Starting Load Balancer ==="
echo ""

# Build backend URLs
BACKEND_URLS=()
for gpu_config in "${GPU_CONFIGS[@]}"; do
    IFS=':' read -r gpu_id port <<< "$gpu_config"
    BACKEND_URLS+=("http://localhost:$port/v1")
done

echo "Load Balancer Configuration:"
echo "  Port: $LB_PORT"
echo "  Strategy: $LB_STRATEGY"
echo "  Backends: ${BACKEND_URLS[*]}"
echo ""

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Check if FastAPI is installed
python3 -c "import fastapi" 2>/dev/null || {
    echo "Error: FastAPI not installed. Install with: pip install fastapi uvicorn httpx"
    exit 1
}

# Start load balancer in background
echo "Starting load balancer..."
nohup python3 -m shared.utils.load_balancer \
    --backends "${BACKEND_URLS[@]}" \
    --host 0.0.0.0 \
    --port "$LB_PORT" \
    --strategy "$LB_STRATEGY" \
    --health-check-interval "$LB_HEALTH_CHECK_INTERVAL" \
    > "$LOG_DIR/load_balancer_port${LB_PORT}.log" 2>&1 &

LB_PID=$!

# Save load balancer PID
echo "$LB_PID" > "$LOG_DIR/vllm_lb_pid.txt"

echo "  -> Load balancer started with PID $LB_PID"
echo "  -> Endpoint: http://localhost:$LB_PORT"
echo "  -> Log: $LOG_DIR/load_balancer_port${LB_PORT}.log"
echo "  -> PID saved to: $LOG_DIR/vllm_lb_pid.txt"

# Wait a bit for load balancer to start
sleep 5

# Check load balancer health
echo ""
echo "Checking load balancer health..."
if curl -s "http://localhost:$LB_PORT/health" > /dev/null 2>&1; then
    echo "  Load balancer: HEALTHY"
    curl -s "http://localhost:$LB_PORT/health" | python3 -m json.tool 2>/dev/null || curl -s "http://localhost:$LB_PORT/health"
else
    echo "  Load balancer: NOT READY (check log: $LOG_DIR/load_balancer_port${LB_PORT}.log)"
fi

echo ""
echo "=========================================="
echo "Deployment Complete!"
echo "=========================================="
echo ""
echo "vLLM Services:"
for i in "${!GPU_CONFIGS[@]}"; do
    gpu_config="${GPU_CONFIGS[$i]}"
    IFS=':' read -r gpu_id port <<< "$gpu_config"
    PID="${VLLM_PIDS[$i]}"
    echo "  GPU $gpu_id: http://localhost:$port/v1 (PID: $PID)"
done
echo ""
echo "Load Balancer:"
echo "  http://localhost:$LB_PORT (PID: $LB_PID)"
echo ""
echo "Configuration:"
echo "  Update llm_service_config.yaml: base_url: \"http://localhost:$LB_PORT/v1\""
echo ""
echo "To stop these specific services, run:"
echo "  ./scripts/stop_vllm_services.sh"
echo ""
echo "This will only kill the processes listed above, not other vLLM services."
echo ""
