"""
Voice Processor Worker - 从处理队列获取任务并执行语音克隆，然后推送到上传队列。
"""

import os
import signal
import time

import requests
import yaml

from services.logger import get_process_worker_logger
from services.queue_manager import QueueManager
from services.voice_processor import VoiceProcessor, AudioTooQuietError

logger = get_process_worker_logger()

should_stop = False


def signal_handler(signum, frame):
    global should_stop
    logger.info("Received shutdown signal, finishing current task...")
    should_stop = True


def load_config(config_path='config.yaml'):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def send_callback_failure(callback_url: str, task_id: str, error: str, error_code: str = None):
    """仅发送失败回调通知"""
    if not callback_url:
        return

    callback_data = {
        'task_uuid': task_id,
        'status': 'failed',
        'timestamp': int(time.time()),
        'error_message': error
    }

    # 如果有错误码，添加到回调数据中
    if error_code:
        callback_data['error_code'] = error_code

    try:
        logger.info(f"Sending FAILURE callback to {callback_url}")
        response = requests.post(callback_url, json=callback_data, timeout=10)
        response.raise_for_status()
        logger.info("Failure callback sent successfully")

    except Exception as e:
        logger.error(f"Failed to send failure callback for task {task_id}: {str(e)}")


def process_single_task(
        task: dict,
        queue_manager: QueueManager,
        voice_processor: VoiceProcessor,
        config: dict
):
    """
    Worker 1 处理单文件任务: 克隆 -> 推送结果
    """
    task_id = task['task_id']
    task_data = task['data']
    hook_url = task_data.get('hook_url')
    output_path = None

    try:
        text = task_data['text']
        language = task_data['language']
        spk_audio_prompt = task_data['spk_audio_prompt']

        logger.info(f"[{task_id}] Processing single task: {language}")

        # 1. 语音处理（临时文件会在 VoiceProcessor 中自动清理）
        output_path = voice_processor.process_single(
            text=text,
            language=language,
            spk_audio_prompt=spk_audio_prompt,
            task_id=task_id,
            logger_instance=logger  # 传递 worker 的 logger 以记录进度
        )

        # 2. 推送到上传队列
        upload_task = {
            'task_id': task_id,
            'hook_url': hook_url,
            'output_paths': [output_path],  # 传递本地路径
            'config': config['task']
        }
        queue_manager.push_upload_task(upload_task)

        logger.info(f"[{task_id}] Single synthesis completed. Pushed to upload queue.")

    except AudioTooQuietError as e:
        error_msg = str(e)
        error_code = e.error_code
        logger.error(f"[{task_id}] Task failed: Audio too quiet. RMS: {e.rms_level:.2f} dB, threshold: {e.threshold} dB")

        # 失败时直接发送回调（包含错误码）
        if hook_url:
            send_callback_failure(hook_url, task_id, error_msg, error_code)

        # 清理可能生成的半成品文件
        if output_path and os.path.exists(output_path):
            os.remove(output_path)
        # 注意：临时文件已在 VoiceProcessor 的 finally 块中自动清理

    except Exception as e:
        error_msg = f"Voice clone failed: {str(e)}"
        logger.error(f"[{task_id}] Task failed: {error_msg}", exc_info=True)

        # 失败时直接发送回调
        if hook_url:
            send_callback_failure(hook_url, task_id, error_msg)

        # 清理可能生成的半成品文件
        if output_path and os.path.exists(output_path):
            os.remove(output_path)
        # 注意：临时文件已在 VoiceProcessor 的 finally 块中自动清理


def main_voice_processor():
    """Voice Processor Worker 主函数"""
    global should_stop

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    config = load_config()

    logger.info("Initializing Voice Processor Worker services...")
    queue_manager = QueueManager(config['redis'])
    voice_processor = VoiceProcessor(config['voice_clone'])

    logger.info("Voice Processor Worker started, waiting for tasks...")

    while not should_stop:
        try:
            # 从处理队列获取任务
            task = queue_manager.get_process_task()
            if task:
                process_single_task(task, queue_manager, voice_processor, config)

        except Exception as e:
            logger.error(f"Voice Processor Worker critical error: {str(e)}", exc_info=True)
            time.sleep(5)

    logger.info("Voice Processor Worker stopped gracefully")


if __name__ == '__main__':
    main_voice_processor()
