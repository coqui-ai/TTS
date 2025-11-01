"""
Voice Processor - 使用 video_tts API 进行语音生成
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional

# 导入 video_tts 的函数
from video_tts import generate_speech

logger = logging.getLogger(__name__)


class AudioTooQuietError(Exception):
    """音频太安静的错误"""

    def __init__(self, message: str, rms_level: float = None, threshold: float = None,
                 error_code: str = "AUDIO_TOO_QUIET"):
        super().__init__(message)
        self.rms_level = rms_level
        self.threshold = threshold
        self.error_code = error_code


def _download_audio_from_url(url: str, output_path: str) -> str:
    """
    从 URL 下载音频文件

    Args:
        url: 音频文件 URL
        output_path: 本地保存路径

    Returns:
        str: 下载后的文件路径
    """
    try:
        import requests

        logger.info(f"Downloading audio from URL: {url}")
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        with open(output_path, 'wb') as f:
            f.write(response.content)

        logger.info(f"Audio downloaded to: {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"Failed to download audio from {url}: {str(e)}")
        raise


class VoiceProcessor:
    """语音处理器 - 使用 video_tts API"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化语音处理器

        Args:
            config: 语音克隆配置字典，包含：
                - output_dir: 输出目录
                - temp_dir: 临时目录
                - device: 设备类型 ('cpu' 或 'cuda' 或 None 表示自动)
        """
        self.config = config
        self.output_dir = Path(config.get('output_dir', 'outputs'))
        self.temp_dir = Path(config.get('temp_dir', 'temp'))
        self.device = config.get('device')

        # 如果 device 为 None，自动检测
        if self.device is None:
            try:
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device = "cpu"

        # 确保目录存在
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"VoiceProcessor initialized: output_dir={self.output_dir}, device={self.device}")

    def _get_audio_sample_path(self, spk_audio_prompt: str, task_id: str) -> str:
        """
        获取参考音频文件路径（支持 URL 和本地路径）
        
        Args:
            spk_audio_prompt: 音频 URL 或本地路径
            task_id: 任务 ID
        
        Returns:
            str: 音频文件路径
        """
        # 如果是 URL，下载到临时目录
        if spk_audio_prompt.startswith(('http://', 'https://')):
            filename = f"{task_id}_reference_{Path(spk_audio_prompt).name}"
            local_path = self.temp_dir / filename

            # 如果文件已存在，直接返回
            if local_path.exists():
                return str(local_path)

            return _download_audio_from_url(spk_audio_prompt, str(local_path))

        # 如果是本地路径，检查是否存在
        local_path = Path(spk_audio_prompt)
        if not local_path.exists():
            raise FileNotFoundError(f"Audio file not found: {spk_audio_prompt}")

        return str(local_path)

    def process_single(
            self,
            text: str,
            language: str,
            spk_audio_prompt: str,
            task_id: str,
            model_name: Optional[str] = None,
            logger_instance: Optional[logging.Logger] = None
    ) -> str:
        """
        处理单个语音克隆任务
        
        Args:
            text: 要合成的文本
            language: 语言代码（如 'en', 'zh', 'ja' 等，直接使用，无需映射）
            spk_audio_prompt: 参考音频 URL 或本地路径
            task_id: 任务 ID
            model_name: 模型名称（可选，如果为 None 则自动选择）
            logger_instance: 日志记录器（可选），如果提供则使用，否则使用模块默认 logger
        
        Returns:
            str: 生成的音频文件路径
        
        Raises:
            AudioTooQuietError: 参考音频太安静
            Exception: 其他处理错误
        
        Note:
            如果参考音频是从 URL 下载的，会在处理完成后自动清理临时文件
        """
        # 使用传入的 logger 或模块默认 logger
        task_logger = logger_instance if logger_instance is not None else logger

        temp_file_to_cleanup = None
        try:
            # 获取参考音频路径
            audio_sample_path = self._get_audio_sample_path(spk_audio_prompt, task_id)

            # 如果是从 URL 下载的文件（在 temp_dir 中），标记需要清理
            if spk_audio_prompt.startswith(('http://', 'https://')):
                temp_file_to_cleanup = audio_sample_path

            # 生成输出路径
            output_filename = f"{task_id}_output.wav"
            output_path = self.output_dir / output_filename

            task_logger.info(f"[{task_id}] Processing: language={language}, text_length={len(text)}")

            # 调用 video_tts 的 generate_speech 方法（直接使用传入的语言代码）
            result_path = generate_speech(
                input_text=text,
                language=language,
                video_sample=audio_sample_path,
                model_name=model_name,
                output_path=str(output_path),
                device=self.device,
                logger=task_logger  # 传递正确的 logger 以记录处理进度
            )

            task_logger.info(f"[{task_id}] Voice generation completed: {result_path}")
            return result_path

        except Exception as e:
            task_logger.error(f"[{task_id}] Voice generation failed: {str(e)}", exc_info=True)
            raise
        finally:
            # 清理临时文件（无论成功还是失败）
            if temp_file_to_cleanup and Path(temp_file_to_cleanup).exists():
                try:
                    Path(temp_file_to_cleanup).unlink()
                    task_logger.info(f"[{task_id}] Cleaned up temporary file: {temp_file_to_cleanup}")
                except Exception as cleanup_error:
                    task_logger.warning(
                        f"[{task_id}] Failed to cleanup temp file {temp_file_to_cleanup}: {str(cleanup_error)}")
