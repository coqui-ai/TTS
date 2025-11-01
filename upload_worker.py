#!/usr/bin/env python3
"""
Uploader Worker - 从上传队列获取任务，上传到 R2 并发送回调。
"""

import os
import signal
import time
from typing import Optional, Any, List

import requests
import yaml

from services.logger import get_upload_worker_logger
from services.queue_manager import QueueManager
from services.r2_uploader import R2Uploader

logger = get_upload_worker_logger()

should_stop = False


def signal_handler(signum, frame):
    global should_stop
    logger.info("Received shutdown signal, finishing current task...")
    should_stop = True


def load_config(config_path='config.yaml'):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def send_callback(callback_url: str, task_id: str, status: str, result: Optional[Any] = None,
                  error: Optional[str] = None, error_code: Optional[str] = None):
    """发送回调通知，使用新的结构"""
    if not callback_url:
        return

    callback_data = {
        'task_uuid': task_id,
        'status': status,
        'timestamp': int(time.time()),
    }

    if status == 'success':
        if isinstance(result, str):
            # 单文件任务
            callback_data['s3_url'] = result
        elif isinstance(result, dict) and 'urls' in result:
            # 批量任务
            callback_data.update(result)

    elif status == 'failed':
        callback_data['error_message'] = error
        # 如果有错误码，添加到回调数据中
        if error_code:
            callback_data['error_code'] = error_code

    MAX_RETRIES = 10
    INITIAL_DELAY = 1

    for attempt in range(MAX_RETRIES):
        try:
            delay = INITIAL_DELAY * (2 ** attempt) + (time.time() * 0.01) % 1

            if attempt > 0:
                logger.info(
                    f"Retrying callback ({status}) for task {task_id} to {callback_url}. Attempt {attempt + 1}/{MAX_RETRIES}. Waiting {delay:.2f}s...")
                time.sleep(min(delay, 60))  # 限制最大等待时间

            logger.info(f"Sending callback ({status}) to {callback_url}")
            response = requests.post(
                callback_url,
                json=callback_data,
                timeout=10
            )
            response.raise_for_status()

            logger.info("Callback sent successfully")
            return

        except requests.exceptions.HTTPError as e:
            if 400 <= e.response.status_code < 500 and e.response.status_code not in [408, 429]:
                logger.error(
                    f"Failed to send callback for task {task_id}: Permanent HTTP error {e.response.status_code}. Details: {str(e)}")
                break

            logger.warning(
                f"Failed to send callback for task {task_id}: Transient HTTP error {e.response.status_code}. Retrying...")

        except requests.exceptions.RequestException as e:
            logger.warning(
                f"Failed to send callback for task {task_id}: Network error or Timeout. Details: {str(e)}. Retrying...")

        except Exception as e:
            logger.error(f"Failed to send callback for task {task_id}: Unexpected error {str(e)}. Giving up.")
            break

    logger.error(f"Failed to send callback for task {task_id} after {MAX_RETRIES} attempts.")


def cleanup_local_files(output_paths: List[str], should_cleanup: bool, task_id: str):
    """清理本地文件"""
    if not should_cleanup:
        return

    cleaned_count = 0
    for output_path in output_paths:
        if os.path.exists(output_path):
            os.remove(output_path)
            cleaned_count += 1

    if cleaned_count > 0:
        logger.info(f"[{task_id}] Cleaned up {cleaned_count} local files.")


def process_upload_task(
        upload_task: dict,
        r2_uploader: R2Uploader,
):
    """
    处理上传任务: 上传到 R2 -> 发送回调
    """
    task_id = upload_task['task_id']
    hook_url = upload_task.get('hook_url')
    output_paths = upload_task['output_paths']
    should_cleanup = upload_task['config'].get('cleanup_after_upload', True)

    try:
        logger.info(f"[{task_id}] Starting R2 upload, count: {len(output_paths)}")

        output_path = output_paths[0]

        # 1. 上传
        file_url = r2_uploader.upload_file(
            output_path,
            object_key=os.path.basename(output_path),
            metadata={'task_id': task_id}
        )

        # 2. 清理
        cleanup_local_files(output_paths, should_cleanup, task_id)

        # 3. 回调
        if hook_url:
            send_callback(hook_url, task_id, 'success', result=file_url)

        logger.info(f"[{task_id}] Upload complete. URL: {file_url}")

    except Exception as e:
        error_msg = f"R2 upload/Callback failed: {str(e)}"
        logger.error(f"[{task_id}] Upload task failed: {error_msg}", exc_info=True)

        # 即使上传失败，如果语音克隆成功，本地文件也应该被清理
        cleanup_local_files(output_paths, should_cleanup, task_id)

        # 发送失败回调
        if hook_url:
            send_callback(hook_url, task_id, 'failed', error=error_msg)


def main_uploader_processor():
    """Uploader Processor Worker 主函数"""
    global should_stop

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    config = load_config()

    logger.info("Initializing Uploader Worker services...")
    queue_manager = QueueManager(config['redis'])
    r2_uploader = R2Uploader(config['r2'])

    logger.info("Uploader Worker started, waiting for tasks in upload queue...")

    while not should_stop:
        try:
            # 从上传队列获取任务 (阻塞等待 5 秒)
            upload_task = queue_manager.get_upload_task(timeout=5)

            if upload_task:
                process_upload_task(upload_task, r2_uploader)

        except Exception as e:
            logger.error(f"Uploader Worker critical error: {str(e)}", exc_info=True)
            time.sleep(5)

    logger.info("Uploader Worker stopped gracefully")


if __name__ == '__main__':
    main_uploader_processor()
