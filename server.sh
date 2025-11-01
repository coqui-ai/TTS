#!/bin/bash
# API 服务器管理脚本 (启动, 停止, 重启)

# 定义主应用文件名
APP_FILE="app.py"

# --- 函数定义 ---

# 停止 API 服务器进程
stop_api() {
    echo "--- Stopping API Server ---"

    # 查找匹配的进程ID
    PIDS=$(pgrep -f "$APP_FILE")

    if [ -z "$PIDS" ]; then
        echo "未找到任何正在运行的 '$APP_FILE' 进程。服务已停止或未启动。"
        return 0 # 成功停止（或本来就没运行）
    else
        echo "找到以下进程 ID (PIDs): $PIDS"

        # 使用 pkill 发送 SIGTERM 信号
        pkill -f "uv run python $APP_FILE"

        if [ $? -eq 0 ]; then
            echo "所有相关服务进程已发送终止信号。"
            # 等待几秒，确认进程是否退出
            sleep 2

            PIDS_AFTER_KILL=$(pgrep -f "$APP_FILE")
            if [ -z "$PIDS_AFTER_KILL" ]; then
                echo "API 服务已成功关闭。"
                return 0
            else
                echo "警告：部分进程在发送终止信号后仍未退出 (PIDs: $PIDS_AFTER_KILL)。可能需要手动 kill -9。"
                return 1 # 警告，未能完全停止
            fi
        else
            echo "错误：尝试停止服务失败。请手动检查进程状态。"
            return 1
        fi
    fi
}

# 启动 API 服务器进程
start_api() {
    echo "--- Starting API Server ---"

    # 检查服务是否已在运行
    if pgrep -f "$APP_FILE" > /dev/null; then
        echo "错误：服务 ($APP_FILE) 似乎已经在运行中。请先停止现有进程。"
        echo "可以使用以下命令查看进程ID：pgrep -f $APP_FILE"
        return 1
    fi

    echo "正在使用 uv run 后台启动 $APP_FILE..."
    # 使用 nohup 和 uv run 启动，并丢弃所有输出
    nohup uv run python "$APP_FILE" > "/dev/null" 2>&1 &

    # 获取后台进程ID (PID)
    PID=$!

    echo "服务已在后台启动！"
    echo "进程 ID (PID): $PID"
    echo "---------------------------"
    echo "停止服务指南: $0 stop 或 kill $PID"
    return 0
}

# 显示使用说明
show_help() {
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  start     Start the API server."
    echo "  stop      Stop the API server."
    echo "  restart   Restart the API server (Stop then Start)."
}

# --- 主逻辑 ---

COMMAND=$1

# 检查命令
if [ -z "$COMMAND" ]; then
    show_help
    exit 1
fi

case "$COMMAND" in
    start)
        start_api
        ;;
    stop)
        stop_api
        ;;
    restart)
        echo "--- RESTARTING API SERVER ---"
        # 1. 停止
        stop_api
        # 确保进程有时间被杀死
        sleep 1
        # 2. 启动
        start_api
        echo "--- RESTART COMPLETE ---"
        ;;
    *)
        echo "Error: Invalid command '$COMMAND'"
        show_help
        exit 1
        ;;
esac

exit 0