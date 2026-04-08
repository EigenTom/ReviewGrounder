#!/bin/bash
# Script to stop reranker services and load balancer (only the ones we started)
# Usage: ./scripts/stop_reranker_services.sh

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

LOG_DIR="./logs/reranker"
PID_FILE="$LOG_DIR/reranker_pids.txt"
LB_PID_FILE="$LOG_DIR/reranker_lb_pid.txt"

echo "=== Stopping Reranker Services and Load Balancer ==="
echo ""

# Step 1: Stop load balancer (if PID file exists)
echo "Step 1: Stopping reranker load balancer..."
if [ -f "$LB_PID_FILE" ]; then
    LB_PID=$(cat "$LB_PID_FILE" 2>/dev/null | head -1)
    if [ -n "$LB_PID" ] && ps -p $LB_PID > /dev/null 2>&1; then
        echo "  Killing load balancer PID $LB_PID..."
        kill -TERM $LB_PID 2>/dev/null || true
        sleep 2
        if ps -p $LB_PID > /dev/null 2>&1; then
            echo "  Force killing load balancer PID $LB_PID..."
            kill -KILL $LB_PID 2>/dev/null || true
        fi
        echo "  Load balancer stopped"
        rm -f "$LB_PID_FILE"
    else
        echo "  Load balancer PID from file not found (may have already terminated)"
        rm -f "$LB_PID_FILE"
    fi
else
    echo "  No load balancer PID file found ($LB_PID_FILE)"
    echo "  If load balancer is running, you may need to find and kill it manually"
fi

echo ""

# Step 2: Stop reranker services (ONLY the ones we started)
echo "Step 2: Stopping reranker services (only the ones we started)..."

if [ -f "$PID_FILE" ]; then
    echo "  Reading PIDs from $PID_FILE"
    KILLED_COUNT=0
    NOT_FOUND_COUNT=0
    
    while IFS= read -r pid || [ -n "$pid" ]; do
        # Skip empty lines
        [ -z "$pid" ] && continue
        
        if ps -p $pid > /dev/null 2>&1; then
            echo "  Killing reranker service PID $pid..."
            kill -TERM $pid 2>/dev/null || true
            KILLED_COUNT=$((KILLED_COUNT + 1))
        else
            echo "  PID $pid: Process not found (may have already terminated)"
            NOT_FOUND_COUNT=$((NOT_FOUND_COUNT + 1))
        fi
    done < "$PID_FILE"
    
    if [ $KILLED_COUNT -gt 0 ]; then
        echo "  Waiting 3 seconds for graceful shutdown..."
        sleep 3
        
        # Force kill if still running
        while IFS= read -r pid || [ -n "$pid" ]; do
            [ -z "$pid" ] && continue
            if ps -p $pid > /dev/null 2>&1; then
                echo "  Force killing reranker service PID $pid..."
                kill -KILL $pid 2>/dev/null || true
            fi
        done < "$PID_FILE"
        
        echo "  Stopped $KILLED_COUNT reranker service(s)"
    else
        echo "  No running processes found from saved PIDs"
    fi
    
    if [ $NOT_FOUND_COUNT -gt 0 ]; then
        echo "  ($NOT_FOUND_COUNT process(es) were already terminated)"
    fi
    
    # Remove PID file after stopping
    rm -f "$PID_FILE"
else
    echo "  WARNING: $PID_FILE not found!"
    echo "  Cannot safely stop services without PID file."
    echo "  If you know the PIDs, you can manually kill them."
    echo "  To avoid affecting other users, DO NOT use pkill!"
fi

echo ""
echo "  NOTE: Only processes from reranker_pids.txt were killed."
echo "  Other reranker services (if any) were NOT affected."

echo ""
echo "=== Checking GPU status ==="
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | grep -E '^ 0,|^ 1,|^ 2,|^ 3,' || echo "GPU 0,1,2,3 status:"

echo ""
echo "Done!"
