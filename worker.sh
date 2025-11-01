#!/bin/bash
# Worker 进程管理脚本 (启动, 停止, 重启)

# --- 默认配置 ---
DEFAULT_PROCESS_WORKERS=1
DEFAULT_UPLOAD_WORKERS=1

# --- 函数定义 ---

# 停止所有 Worker 进程
stop_workers() {
    echo "--- Stopping Workers ---"
    # 查找并杀死所有 uv run python *_worker.py 进程
    pkill -f "uv run python.*worker.py"

    if [ $? -eq 0 ]; then
        echo "All workers stopped successfully!"
    else
        echo "No worker processes found or failed to stop workers (or process not running via pkill signature)."
    fi
}

# 启动 Worker 进程
start_workers() {
    local PROCESS_WORKERS=$1
    local UPLOAD_WORKERS=$2

    echo "--- Worker Configuration ---"
    echo "Starting $PROCESS_WORKERS Process Worker(s)..."
    echo "Starting $UPLOAD_WORKERS Upload Worker(s)..."
    echo "----------------------------"

    # --- 1. 启动 Process Workers ---
    for i in $(seq 1 $PROCESS_WORKERS); do
        # 在后台运行，日志输出到 /dev/null
        nohup uv run python process_worker.py > '/dev/null' 2>&1 &
        WORKER_PID=$!
        echo "Process Worker $i started with PID: $WORKER_PID"
    done

    # --- 2. 启动 Upload Workers ---
    for i in $(seq 1 $UPLOAD_WORKERS); do
        # 在后台运行，日志输出到 /dev/null
        nohup uv run python upload_worker.py > '/dev/null' 2>&1 &
        WORKER_PID=$!
        echo "Upload Worker $i started with PID: $WORKER_PID"
    done

    echo "All workers started successfully!"
    echo "Total workers: $((PROCESS_WORKERS + UPLOAD_WORKERS))"
}

# 显示使用说明
show_help() {
    echo "Usage: $0 [command] [process_workers] [upload_workers]"
    echo ""
    echo "Commands:"
    echo "  start     Start workers."
    echo "            Optional: process_workers (default: $DEFAULT_PROCESS_WORKERS), upload_workers (default: $DEFAULT_UPLOAD_WORKERS)"
    echo "  stop      Stop all running workers."
    echo "  restart   Restart all workers (Stop then Start)."
    echo ""
    echo "Examples:"
    echo "  $0 start 4 2    # Start 4 Process Workers and 2 Upload Workers"
    echo "  $0 restart      # Restart with default counts ($DEFAULT_PROCESS_WORKERS/$DEFAULT_UPLOAD_WORKERS)"
}

# --- 主逻辑 ---

COMMAND=$1

# 检查命令
if [ -z "$COMMAND" ]; then
    show_help
    exit 1
fi

# 确定 Worker 数量 (用于 start 和 restart)
P_COUNT=${2:-$DEFAULT_PROCESS_WORKERS}
U_COUNT=${3:-$DEFAULT_UPLOAD_WORKERS}

case "$COMMAND" in
    start)
        start_workers "$P_COUNT" "$U_COUNT"
        ;;
    stop)
        stop_workers
        ;;
    restart)
        echo "--- RESTARTING WORKERS ---"
        stop_workers
        # 确保进程有时间被杀死，等待 1 秒
        sleep 1
        start_workers "$P_COUNT" "$U_COUNT"
        echo "--- RESTART COMPLETE ---"
        ;;
    *)
        echo "Error: Invalid command '$COMMAND'"
        show_help
        exit 1
        ;;
esac

exit 0