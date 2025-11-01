#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TTS 语音生成工具

从文本文件读取内容，使用 TTS 模型生成指定语言的语音。
支持从视频样本中提取音频作为参考进行语音克隆。
"""

import argparse
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Optional, Any

try:
    from moviepy.editor import VideoFileClip
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False

try:
    import ffmpeg
    FFMPEG_AVAILABLE = True
except ImportError:
    FFMPEG_AVAILABLE = False

from TTS.api import TTS


def get_reference_audio(audio_or_video_path: str, output_audio_path: str, logger: Optional[Any] = None) -> str:
    """
    获取参考音频，支持直接音频文件或从视频中提取
    
    Args:
        audio_or_video_path: 音频文件或视频文件路径
        output_audio_path: 输出音频文件路径（仅当需要从视频提取时使用）
    
    Returns:
        参考音频文件路径
    """
    path = Path(audio_or_video_path)
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {path}")
    
    # 支持的音频格式
    audio_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.aac', '.ogg', '.opus'}
    # 常见的视频格式
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.m4v'}
    
    file_ext = path.suffix.lower()
    
    # 定义输出函数
    def log_info(msg: str):
        if logger:
            logger.info(msg)
        else:
            print(msg)
    
    # 如果是音频文件，直接返回
    if file_ext in audio_extensions:
        log_info(f"🎵 检测到音频文件，直接使用: {path}")
        return str(path)
    
    # 如果是视频文件，需要提取音频
    if file_ext in video_extensions:
        log_info(f"📹 检测到视频文件，正在提取音频: {path}")
        if MOVIEPY_AVAILABLE:
            video = VideoFileClip(str(path))
            audio = video.audio
            if audio is None:
                video.close()
                raise ValueError("视频文件中没有音频轨道")
            audio.write_audiofile(output_audio_path, verbose=False, logger=None)
            video.close()
            audio.close()
            return output_audio_path
        elif FFMPEG_AVAILABLE:
            try:
                (
                    ffmpeg
                    .input(str(path))
                    .output(output_audio_path, acodec='pcm_s16le', ac=1, ar='22050')
                    .overwrite_output()
                    .run(capture_stdout=True, capture_stderr=True, quiet=True)
                )
                return output_audio_path
            except ffmpeg.Error as e:
                raise RuntimeError(f"FFmpeg 错误: {e.stderr.decode()}")
        else:
            raise ImportError("需要安装 moviepy 或 ffmpeg-python 来从视频提取音频")
    
    # 未知格式，尝试作为音频文件处理
    log_info(f"⚠️  未知文件格式 ({file_ext})，尝试作为音频文件使用: {path}")
    return str(path)


def get_text_from_input(input_str: str) -> str:
    """
    从输入获取文本，支持文件路径或直接文本
    
    Args:
        input_str: 文件路径或直接文本内容
    
    Returns:
        文本内容
    """
    input_path = Path(input_str)
    
    # 如果输入是存在的文件路径，则从文件读取
    if input_path.exists() and input_path.is_file():
        # 注意：get_text_from_input 没有 logger 参数，保持 print
        print(f"📄 检测到文件路径，正在读取: {input_path}")
        for encoding in ('utf-8', 'gbk'):
            try:
                with open(input_path, 'r', encoding=encoding) as f:
                    text = f.read().strip()
                if not text:
                    raise ValueError(f"文本文件为空: {input_path}")
                return text
            except UnicodeDecodeError:
                continue
        raise ValueError(f"无法读取文本文件，请确保文件使用 UTF-8 或 GBK 编码: {input_path}")
    else:
        # 否则直接作为文本使用
        text = input_str.strip()
        if not text:
            raise ValueError("输入的文本不能为空")
        return text


def select_model(language: str) -> str:
    """根据语言自动选择模型"""
    lang_lower = language.lower()
    model_map = {
        'zh': "tts_models/zh-CN/baker/tacotron2-DDC-GST",
        'chinese': "tts_models/zh-CN/baker/tacotron2-DDC-GST",
        'cn': "tts_models/zh-CN/baker/tacotron2-DDC-GST",
        'en': "tts_models/en/ljspeech/tacotron2-DDC",
        'english': "tts_models/en/ljspeech/tacotron2-DDC",
        'fr': "tts_models/fr/mai/tacotron2-DDC",
        'french': "tts_models/fr/mai/tacotron2-DDC",
        'de': "tts_models/de/thorsten/tacotron2-DDC",
        'german': "tts_models/de/thorsten/tacotron2-DDC",
    }
    return model_map.get(lang_lower, "tts_models/multilingual/multi-dataset/xtts_v2")


def generate_speech(
    input_text: str,
    language: str,
    video_sample: Optional[str] = None,
    model_name: Optional[str] = None,
    output_path: Optional[str] = None,
    device: str = "cpu",
    logger: Optional[Any] = None
) -> str:
    """
    从文本生成语音（支持文件路径或直接文本）
    
    Args:
        input_text: 输入文本或文本文件路径
        language: 目标语言代码（如 'en', 'zh', 'fr' 等）
        video_sample: 视频样本路径（用于提取参考音频进行语音克隆）
        model_name: TTS 模型名称，如果为 None 则自动选择
        output_path: 输出音频文件路径，如果为 None 则自动生成
        device: 设备类型，'cpu' 或 'cuda'
    logger: 日志记录器（可选），如果提供则使用日志输出，否则使用 print
    
    Returns:
        输出音频文件路径
    """
    # 定义输出函数：如果有 logger 则使用 logger，否则使用 print
    def log_info(msg: str):
        if logger:
            logger.info(msg)
        else:
            print(msg)
    
    def log_error(msg: str):
        if logger:
            logger.error(msg)
        else:
            print(msg)
    
    def log_warning(msg: str):
        if logger:
            logger.warning(msg)
        else:
            print(msg)
    
    # 获取文本内容（自动识别是文件还是直接文本）
    try:
        text = get_text_from_input(input_text)
        log_info(f"✅ 文本准备完成，共 {len(text)} 个字符")
    except Exception as e:
        log_error(f"❌ 获取文本失败: {e}")
        raise
    
    # 确定输出路径
    if output_path is None:
        # 如果输入是文件路径，基于文件名生成输出
        input_path = Path(input_text)
        if input_path.exists() and input_path.is_file():
            output_path = str(input_path.parent / f"{input_path.stem}_tts_{language}.wav")
        else:
            # 如果是直接文本，生成默认文件名
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"output_tts_{language}_{timestamp}.wav"
    else:
        output_path = str(Path(output_path))
    
    # 获取参考音频（支持直接音频文件或从视频提取）
    reference_audio = None
    if video_sample:
        # 使用临时文件保存提取的音频（如果需要），确保在 TTS 使用前不被删除
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_audio:
            extracted_audio_path = tmp_audio.name
        
        try:
            reference_audio = get_reference_audio(video_sample, extracted_audio_path, logger=logger)
            log_info(f"✅ 参考音频准备完成")
        except Exception as e:
            log_error(f"❌ 获取参考音频失败: {e}")
            # 清理临时文件
            Path(extracted_audio_path).unlink(missing_ok=True)
            raise
    
    # 初始化 TTS 模型
    if model_name is None:
        model_name = select_model(language)
    
    log_info(f"🤖 正在初始化 TTS 模型: {model_name}")
    try:
        tts = TTS(model_name=model_name, progress_bar=True)
        log_info(f"📥 模型加载中...")
        tts.to(device)
        log_info(f"📦 模型已移动到设备: {device}")
    except Exception as e:
        log_error(f"❌ TTS 模型加载失败: {e}")
        log_warning("💡 提示: 尝试使用其他模型或检查网络连接")
        # 清理临时文件
        if reference_audio:
            Path(reference_audio).unlink(missing_ok=True)
        raise
    
    log_info(f"✅ TTS 模型加载完成")
    
    # 生成语音
    log_info(f"🎤 正在生成语音 (语言: {language}, 文本长度: {len(text)} 字符)...")
    
    import time
    start_time = time.time()
    
    try:
        # 简化逻辑：统一调用 tts_to_file，让它自己处理参数
        kwargs = {
            'text': text,
            'file_path': output_path
        }
        
        # 如果是多语言模型，添加语言参数
        is_multilingual = ("xtts" in model_name.lower() or 
                         "your_tts" in model_name.lower() or
                         (hasattr(tts, 'is_multi_lingual') and tts.is_multi_lingual))
        
        if is_multilingual:
            kwargs['language'] = language
            log_info(f"🌐 使用多语言模型，语言: {language}")
        
        # 如果提供了参考音频，添加 speaker_wav 参数
        if reference_audio:
            kwargs['speaker_wav'] = reference_audio
            log_info(f"🎯 使用参考音频进行语音克隆: {reference_audio}")
        
        log_info(f"🔄 开始语音合成...")
        tts.tts_to_file(**kwargs)
        
        elapsed_time = time.time() - start_time
        log_info(f"⏱️  语音合成耗时: {elapsed_time:.2f} 秒")
        
    except Exception as e:
        log_error(f"❌ 语音生成失败: {e}")
        raise
    finally:
        # 清理临时文件（仅当是从视频提取的音频时）
        if reference_audio and Path(reference_audio).exists():
            # 检查是否是临时文件（通过路径是否包含临时目录特征判断）
            try:
                if 'tmp' in reference_audio.lower() or tempfile.gettempdir() in reference_audio:
                    Path(reference_audio).unlink(missing_ok=True)
            except Exception:
                pass  # 忽略清理错误
    
    log_info(f"✅ 语音生成完成: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="TTS 语音生成工具 - 从文本或文本文件生成指定语言的语音",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本使用：直接输入文本
  python video_tts.py --input "Hello world" --language en --output output.wav
  
  # 从文本文件生成语音
  python video_tts.py --input text.txt --language en --output output.wav
  
  # 使用音频/视频样本进行语音克隆（直接文本）
  uv run video_tts.py --input "恭喜恭喜" --language zh --video-sample audio.wav --output output.wav
  uv run video_tts.py --input "恭喜恭喜" --language zh --video-sample sample.mp4 --output output.wav
  
  # 使用指定的 TTS 模型
  python video_tts.py --input "Hello world" --language en --model tts_models/multilingual/multi-dataset/xtts_v2
  
  # 使用 GPU 加速
  python video_tts.py --input "Hello world" --language en --device cuda --output output.wav
        """
    )
    
    parser.add_argument("--input", "-i", type=str, required=True, help="输入文本或文本文件路径（如果路径存在则读取文件，否则作为文本使用）")
    parser.add_argument("--language", "-l", type=str, required=True, help="目标语言代码 (如: en, zh, fr, de 等)")
    parser.add_argument("--video-sample", "-v", type=str, default=None, help="音频或视频样本路径（可选，用于语音克隆。支持音频文件：.wav, .mp3, .flac 等；视频文件：.mp4, .avi, .mov 等）")
    parser.add_argument("--model", "-m", type=str, default=None, help="TTS 模型名称，如果未指定则根据语言自动选择")
    parser.add_argument("--output", "-o", type=str, default=None, help="输出音频文件路径（.wav），如果未指定则自动生成")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="设备类型 (cpu 或 cuda)")
    
    args = parser.parse_args()
    
    try:
        output_path = generate_speech(
            input_text=args.input,
            language=args.language,
            video_sample=args.video_sample,
            model_name=args.model,
            output_path=args.output,
            device=args.device
        )
        
        print(f"\n🎉 处理完成！")
        print(f"📁 输出文件: {output_path}")
        sys.exit(0)
    
    except KeyboardInterrupt:
        print("\n⚠️  用户中断操作")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
