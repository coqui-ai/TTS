# TTS 语音生成工具使用说明

## 功能概述

`video_tts.py` 提供了一个命令行工具和 Python API，用于从文本文件生成语音：
1. 从文本文件读取内容
2. 从视频样本中提取音频作为参考（可选，用于语音克隆）
3. 使用 TTS（文本转语音）模型生成指定语言的语音
4. 输出为音频文件（.wav）

## 安装依赖

```bash
# 如果需要从视频样本提取参考音频（可选）
pip install moviepy
# 或
pip install ffmpeg-python

# TTS 模型会在首次使用时自动下载
```

## 使用方法

### 1. 命令行使用

#### 基本用法（仅文本转语音）
```bash
python video_tts.py \
    --input text.txt \
    --language en \
    --output output.wav
```

#### 使用视频样本进行语音克隆
```bash
python video_tts.py \
    --input text.txt \
    --language zh \
    --video-sample sample.mp4 \
    --output output.wav
```

#### 使用指定的 TTS 模型
```bash
python video_tts.py \
    --input text.txt \
    --language en \
    --model tts_models/multilingual/multi-dataset/xtts_v2 \
    --output output.wav
```

#### 使用 GPU 加速
```bash
python video_tts.py \
    --input text.txt \
    --language en \
    --device cuda \
    --output output.wav
```

### 2. Python API 使用

```python
from video_tts import generate_speech

# 基本使用：仅文本转语音
output_path = generate_speech(
    text_file="input.txt",
    language="en",
    output_path="output.wav",
    device="cpu"
)

# 使用视频样本进行语音克隆
output_path = generate_speech(
    text_file="input.txt",
    language="zh",
    video_sample="sample.mp4",
    output_path="output.wav",
    device="cuda"
)

# 使用指定的模型
output_path = generate_speech(
    text_file="input.txt",
    language="fr",
    video_sample="sample.mp4",
    model_name="tts_models/multilingual/multi-dataset/xtts_v2",
    output_path="output.wav",
    device="cpu"
)
```

## 参数说明

### 命令行参数

- `--input, -i`: 输入文本文件路径（必需）
- `--language, -l`: 目标语言代码（必需），如: `en`, `zh`, `fr`, `de`, `es` 等
- `--video-sample, -v`: 视频样本文件路径（可选），用于提取参考音频进行语音克隆
- `--model, -m`: TTS 模型名称（可选），如果未指定则根据语言自动选择
- `--output, -o`: 输出音频文件路径（可选），默认自动生成（.wav 格式）
- `--device`: 设备类型，`cpu` 或 `cuda`（默认：`cpu`）

### Python API 参数

```python
generate_speech(
    text_file: str,                # 输入文本文件路径
    language: str,                 # 目标语言代码
    video_sample: Optional[str],   # 视频样本文件路径（可选）
    model_name: Optional[str],      # TTS 模型名称（可选）
    output_path: Optional[str],    # 输出音频文件路径（可选）
    device: str                    # 设备类型：'cpu' 或 'cuda'
)
```

## 文本文件格式

- 支持 UTF-8 编码（推荐）
- 支持 GBK 编码（中文环境）
- 文件内容会被读取为纯文本
- 会自动去除首尾空白字符

## 支持的模型

### 自动模型选择

根据语言代码自动选择：
- **中文 (zh)**: `tts_models/zh-CN/baker/tacotron2-DDC-GST`
- **英文 (en)**: `tts_models/en/ljspeech/tacotron2-DDC`
- **法语 (fr)**: `tts_models/fr/mai/tacotron2-DDC`
- **德语 (de)**: `tts_models/de/thorsten/tacotron2-DDC`
- **其他语言**: `tts_models/multilingual/multi-dataset/xtts_v2`（多语言模型，支持语音克隆）

### 推荐模型

对于语音克隆和多语言任务，推荐使用：
- `tts_models/multilingual/multi-dataset/xtts_v2` - 支持 16+ 语言，支持语音克隆
- `tts_models/multilingual/multi-dataset/your_tts` - 支持多语言语音克隆

## 工作流程

1. **读取文本**: 从指定的文本文件中读取内容
2. **提取参考音频**（如果提供了视频样本）: 从视频样本中提取音频轨道作为参考
3. **初始化 TTS**: 加载指定的 TTS 模型（或根据语言自动选择）
4. **生成语音**: 根据文本和目标语言生成语音
   - 如果提供了视频样本，使用提取的参考音频进行语音克隆
   - 如果未提供视频样本，使用模型的默认语音
5. **保存输出**: 将生成的语音保存为音频文件（.wav 格式）

## 使用示例

### 示例 1: 基本文本转语音

创建文本文件 `hello.txt`:
```
Hello, this is a test. This is generated speech from text.
```

运行命令:
```bash
python video_tts.py --input hello.txt --language en --output hello.wav
```

### 示例 2: 使用视频样本进行语音克隆

创建文本文件 `chinese.txt`:
```
你好，这是使用语音克隆技术生成的语音。
```

运行命令:
```bash
python video_tts.py \
    --input chinese.txt \
    --language zh \
    --video-sample reference.mp4 \
    --output chinese_cloned.wav
```

### 示例 3: 批量处理多个文本文件

```python
from pathlib import Path
from video_tts import generate_speech

text_files = Path("texts").glob("*.txt")
for text_file in text_files:
    output = generate_speech(
        text_file=str(text_file),
        language="en",
        output_path=f"outputs/{text_file.stem}.wav"
    )
    print(f"Generated: {output}")
```

## 注意事项

1. **依赖要求**:
   - 如果需要从视频样本提取音频，需要安装 `moviepy` 或 `ffmpeg-python`
   - 确保已安装 TTS 项目的所有依赖

2. **模型下载**:
   - 首次使用某个模型时会自动下载
   - 下载的模型会缓存在本地

3. **性能考虑**:
   - GPU 加速可以显著提高语音生成速度
   - 文本长度会影响生成时间

4. **语言支持**:
   - 不同模型支持的语言不同
   - 多语言模型（如 XTTS）支持更多语言

5. **视频样本要求**:
   - 视频文件必须包含音频轨道
   - 支持常见的视频格式（mp4, avi, mov 等）
   - 视频中的音频将用作语音克隆的参考

6. **文本文件编码**:
   - 推荐使用 UTF-8 编码
   - 如果使用中文，GBK 编码也被支持

## 故障排除

### 问题 1: 找不到视频处理库（仅在使用视频样本时）
**错误**: `需要安装 moviepy 或 ffmpeg-python`
**解决**: 
```bash
pip install moviepy
# 或
pip install ffmpeg-python
```

### 问题 2: 模型下载失败
**错误**: 网络问题导致模型下载失败
**解决**: 检查网络连接，或手动下载模型

### 问题 3: 语音生成失败
**错误**: `语音生成失败`
**解决**: 
- 检查文本文件是否为空
- 确认模型支持目标语言
- 检查是否有足够的 GPU 内存（如果使用 CUDA）

### 问题 4: 无法读取文本文件
**错误**: `无法读取文本文件`
**解决**: 
- 确认文件使用 UTF-8 或 GBK 编码
- 检查文件是否存在且可读
- 确认文件内容不为空

### 问题 5: 视频样本处理失败（仅在使用视频样本时）
**错误**: `提取参考音频失败`
**解决**: 
- 确认视频文件格式受支持
- 检查视频文件是否包含音频轨道
- 确认视频文件未损坏

## 相关文件

- `video_tts.py` - 主功能模块（位于项目根目录，与 uv.lock 同级）
- `TTS/api.py` - TTS API 接口

## 更多信息

查看 TTS 项目的完整文档：
- [TTS 文档](https://tts.readthedocs.io/)
- [模型列表](https://github.com/coqui-ai/TTS/wiki)
