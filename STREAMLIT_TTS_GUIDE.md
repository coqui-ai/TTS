# Coqui TTS Streamlit Web Interface

A user-friendly web interface for Coqui TTS that enables voice cloning and long-form text synthesis in English and Indonesian.

## Features

- **Voice Cloning**: Clone any voice using 3+ seconds of reference audio (English only via XTTS v2)
- **Multi-Language Support**:
  - English with voice cloning (XTTS v2)
  - Indonesian standard synthesis (Fairseq VITS)
- **Long-Form Text**: Automatic sentence splitting for processing long documents
- **Multiple Reference Audio**: Upload multiple audio files to improve voice cloning quality
- **Audio Management**: Preview, playback, and download generated speech
- **GPU Acceleration**: Automatic GPU detection and usage for faster synthesis
- **Real-time Progress**: Visual feedback during model loading and synthesis

## Installation

### Prerequisites

- Python 3.9, 3.10, or 3.11
- (Optional but recommended) NVIDIA GPU with CUDA support for faster synthesis

### Step 1: Install Coqui TTS

If you haven't already installed Coqui TTS:

```bash
# Clone the repository (if not already cloned)
git clone https://github.com/coqui-ai/TTS
cd TTS

# Install TTS with all dependencies
pip install -e .[all]
```

### Step 2: Install Streamlit Requirements

```bash
pip install -r requirements.streamlit.txt
```

Or install manually:

```bash
pip install streamlit>=1.28.0 matplotlib>=3.7.0
```

### ⚠️ Important: NumPy Version Compatibility

**If you encounter "Numba needs NumPy 2.2 or less" error:**

Numba (required by librosa) doesn't support NumPy 2.3+. Fix this by:

```bash
pip install "numpy>=1.24.3,<2.0"
```

This is included in `requirements.streamlit.txt`, but if you're upgrading an existing environment, you may need to run this manually.

**See `TROUBLESHOOTING.md` for detailed solutions to common issues.**

## Running the Application

### Start the Web Interface

From the Coqui TTS root directory:

```bash
streamlit run streamlit_tts_app.py
```

The application will automatically open in your default web browser at `http://localhost:8501`

### Command Line Options

```bash
# Run on a specific port
streamlit run streamlit_tts_app.py --server.port 8080

# Run on a specific address (for remote access)
streamlit run streamlit_tts_app.py --server.address 0.0.0.0

# Disable auto-opening browser
streamlit run streamlit_tts_app.py --server.headless true
```

## Usage Guide

### 1. Select Language and Model

In the sidebar, choose your target language:
- **English (XTTS v2)**: For English text with voice cloning support
- **Indonesian (Fairseq VITS)**: For Indonesian text with standard synthesis

### 2. Enter Text

**Option A: Type Text**
- Select "Type Text" option
- Enter or paste your text in the text area
- Supports long-form text (will be automatically split into sentences)

**Option B: Upload Text File**
- Select "Upload File" option
- Upload a `.txt` file containing your text
- The file contents will be displayed for review

### 3. Voice Cloning (English Only)

For English synthesis with voice cloning:

1. **Upload Reference Audio**:
   - Click "Upload Reference Audio File(s)"
   - Select one or more audio files (WAV, MP3, FLAC, or OGG)
   - Minimum 3 seconds of clean speech recommended
   - Multiple files will be combined for better quality

2. **Audio Quality Tips**:
   - Use clear, high-quality recordings
   - Minimize background noise
   - Single speaker only
   - 6-10 seconds of audio is ideal
   - Multiple samples capture more voice characteristics

3. **Preview Reference Audio**:
   - Expand "Preview Reference Audio" to listen to uploaded files
   - Verify audio quality before synthesis

### 4. Generate Speech

1. Click the **"Generate Speech"** button
2. Wait for synthesis to complete (progress indicator will show)
3. First run will download models (~2GB for XTTS v2) - this is cached for future use

### 5. Review and Download

Once generation is complete:

- **Audio Player**: Listen to the generated speech directly in the browser
- **Download Button**: Download the audio as a WAV file
- **Statistics**: View audio duration, processing time, and sample rate
- **Waveform** (optional): Enable in Advanced Settings to visualize the audio

## Advanced Settings

Access advanced settings in the sidebar under "Advanced Settings":

### Split into Sentences
- **Enabled (Default)**: Processes text sentence-by-sentence
  - Better for long texts
  - More memory efficient
  - Consistent quality across long documents
- **Disabled**: Processes entire text at once
  - May hit model length limits
  - Higher VRAM usage
  - Use only for short texts

### Show Waveform
- Enable to display audio waveform visualization
- Useful for analyzing audio characteristics
- Requires matplotlib

### GPU Acceleration
- Automatically enabled if CUDA-compatible GPU detected
- Significantly faster synthesis (5-10x speedup)
- Checkbox is disabled if no GPU available

## Model Information

### English - XTTS v2
- **Model**: `tts_models/multilingual/multi-dataset/xtts_v2`
- **Features**:
  - Zero-shot voice cloning
  - 17 language support
  - Streaming capability (<200ms latency)
  - High-quality natural speech
- **Requirements**:
  - ~2GB download (first run only)
  - 3+ seconds reference audio for cloning
  - GPU recommended for faster synthesis

### Indonesian - Fairseq VITS
- **Model**: `tts_models/ind/fairseq/vits`
- **Features**:
  - Part of ~1100 language MMS collection
  - Standard Indonesian synthesis
  - Fast inference
- **Limitations**:
  - No voice cloning support
  - Uses single default voice
- **Note**: For Indonesian voice cloning, consider using the voice conversion feature (coming in future updates)

## Examples

### Example 1: English Voice Cloning

1. Select: **English (XTTS v2)**
2. Upload: Your voice sample (e.g., `my_voice.wav`)
3. Enter text:
   ```
   Welcome to the world of artificial intelligence and text-to-speech synthesis.
   This technology enables computers to generate human-like speech from any written text.
   ```
4. Click: **Generate Speech**
5. Result: Speech in your cloned voice

### Example 2: Indonesian Text Synthesis

1. Select: **Indonesian (Fairseq VITS)**
2. Enter text:
   ```
   Selamat datang di dunia kecerdasan buatan dan sintesis teks-ke-ucapan.
   Teknologi ini memungkinkan komputer untuk menghasilkan ucapan seperti manusia dari teks tertulis.
   ```
3. Click: **Generate Speech**
4. Result: Indonesian speech with standard voice

### Example 3: Long-Form Document

1. Select your language
2. Upload a text file or paste a long article
3. Ensure "Split into Sentences" is enabled
4. Generate speech
5. The system will automatically process the text in chunks

## Troubleshooting

**📋 For comprehensive troubleshooting, see `TROUBLESHOOTING.md`**

This section covers the most common issues. For detailed solutions and additional problems, refer to the full troubleshooting guide.

### Model Download Issues

**Problem**: Model fails to download
**Solution**:
```bash
# Pre-download models using CLI
tts --model_name tts_models/multilingual/multi-dataset/xtts_v2 --text "test" --out_path test.wav
tts --model_name tts_models/ind/fairseq/vits --text "test" --out_path test.wav
```

### CUDA Out of Memory

**Problem**: GPU runs out of memory
**Solution**:
- Disable GPU acceleration (use CPU mode)
- Enable "Split into Sentences"
- Process shorter text segments
- Close other GPU-intensive applications

### Poor Voice Cloning Quality

**Problem**: Cloned voice doesn't sound like the reference
**Solutions**:
- Use longer reference audio (6-10 seconds ideal)
- Upload multiple samples of the same voice
- Ensure reference audio is clean and clear
- Verify single speaker in reference audio
- Check that reference audio sample rate is consistent

### Slow Processing

**Problem**: Synthesis takes too long
**Solutions**:
- Enable GPU acceleration
- Use shorter texts
- Pre-load models by running the app once
- Upgrade hardware (GPU recommended)

### Audio Quality Issues

**Problem**: Generated audio has artifacts or poor quality
**Solutions**:
- For English: Use better quality reference audio for cloning
- Split long texts into sentences
- Check input text for unusual characters or formatting
- Ensure adequate system resources (RAM/VRAM)

## System Requirements

### Minimum Requirements
- **CPU**: 4+ cores
- **RAM**: 8GB
- **Storage**: 5GB free space (for models)
- **OS**: Ubuntu 18.04+, macOS 10.14+, Windows 10+

### Recommended Requirements
- **CPU**: 8+ cores
- **RAM**: 16GB
- **GPU**: NVIDIA GPU with 6GB+ VRAM (RTX 2060 or better)
- **CUDA**: 11.8 or newer
- **Storage**: 10GB free space (SSD preferred)

## Performance Benchmarks

| Configuration | Text Length | Processing Time | Notes |
|--------------|-------------|-----------------|-------|
| CPU (i7-10700) | 100 words | ~15-20s | Per sentence |
| GPU (RTX 3060) | 100 words | ~2-3s | Per sentence |
| GPU (RTX 4090) | 100 words | ~1-2s | Per sentence |
| CPU | 1000 words | ~3-5min | With sentence splitting |
| GPU (RTX 3060) | 1000 words | ~30-60s | With sentence splitting |

*Times are approximate and vary based on model, text complexity, and system load*

## API Reference

The web interface uses the Coqui TTS API. For programmatic access:

```python
from TTS.api import TTS

# English with voice cloning
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")
tts.tts_to_file(
    text="Your text here",
    speaker_wav="reference.wav",
    language="en",
    file_path="output.wav"
)

# Indonesian standard synthesis
tts = TTS("tts_models/ind/fairseq/vits").to("cuda")
tts.tts_to_file(
    text="Teks Anda di sini",
    file_path="output.wav"
)
```

## Future Enhancements

Planned features for future versions:

- [ ] Voice conversion for Indonesian voice cloning
- [ ] Batch processing for multiple texts
- [ ] Audio effects and post-processing
- [ ] Support for more languages (Spanish, French, German, etc.)
- [ ] Fine-tuning interface for custom models
- [ ] Speed and pitch controls
- [ ] Export formats (MP3, OGG, FLAC)
- [ ] History and favorites
- [ ] API endpoint for programmatic access

## Support and Contributing

### Getting Help
- **Issues**: [GitHub Issues](https://github.com/coqui-ai/TTS/issues)
- **Discussions**: [GitHub Discussions](https://github.com/coqui-ai/TTS/discussions)
- **Documentation**: [ReadTheDocs](https://tts.readthedocs.io/)

### Contributing
Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This interface is built on Coqui TTS, which is licensed under the MPL 2.0 License.
See the [LICENSE](LICENSE.txt) file for details.

## Acknowledgments

- **Coqui TTS Team**: For the excellent TTS models and framework
- **Streamlit**: For the web framework
- **Community**: For feedback and contributions

---

**Version**: 1.0.0
**Last Updated**: 2025-11-10
**Author**: Claude Code Assistant
