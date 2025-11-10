# Troubleshooting Guide - Coqui TTS Streamlit Interface

## Common Issues and Solutions

### ❌ Error: "Numba needs NumPy 2.2 or less. Got NumPy 2.3"

**Problem:**
```
ImportError: Numba needs NumPy 2.2 or less. Got NumPy 2.3.
```

**Cause:**
Your NumPy version is too new. Numba (required by librosa, which is used by TTS) doesn't support NumPy 2.3+.

**Solution:**

**Option 1: Downgrade NumPy (Quick Fix)**
```bash
pip install "numpy>=1.24.3,<2.0"
```

**Option 2: Reinstall from Requirements**
```bash
pip install -r requirements.txt --force-reinstall
```

**Option 3: Fresh Virtual Environment (Recommended)**
```bash
# Create new virtual environment
python3 -m venv tts_env
source tts_env/bin/activate  # On Windows: tts_env\Scripts\activate

# Install TTS
pip install -e .[all]

# Install Streamlit
pip install -r requirements.streamlit.txt
```

**Verification:**
```bash
python3 -c "import numpy; print(f'NumPy version: {numpy.__version__}')"
# Should show version 1.x.x (not 2.3+)
```

---

### ❌ Error: "No module named 'streamlit'"

**Problem:**
```
ModuleNotFoundError: No module named 'streamlit'
```

**Solution:**
```bash
pip install streamlit
# or
pip install -r requirements.streamlit.txt
```

---

### ❌ Error: "CUDA out of memory"

**Problem:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**

**1. Disable GPU Mode**
- In the Streamlit sidebar, uncheck "Use GPU Acceleration"

**2. Enable Sentence Splitting**
- In Advanced Settings, ensure "Split into Sentences" is enabled

**3. Process Shorter Texts**
- Break your text into smaller chunks
- Process each chunk separately

**4. Close Other Applications**
- Close other GPU-intensive applications
- Free up VRAM

**5. Use CPU Mode Explicitly**
```python
# If running programmatically
tts = TTS("model_name", gpu=False)
```

---

### ❌ Error: "Model download failed"

**Problem:**
Models fail to download or timeout.

**Solutions:**

**1. Manual Download**
```bash
# Pre-download English model
tts --model_name tts_models/multilingual/multi-dataset/xtts_v2 \
    --text "test" --out_path test.wav

# Pre-download Indonesian model
tts --model_name tts_models/ind/fairseq/vits \
    --text "test" --out_path test.wav
```

**2. Check Internet Connection**
- Ensure stable internet connection
- Models are ~2GB and require good bandwidth

**3. Check Disk Space**
- Ensure at least 5GB free space
- Models are cached in `~/.local/share/tts/`

**4. Clear Cache and Retry**
```bash
rm -rf ~/.local/share/tts/
# Then retry running the app
```

---

### ❌ Error: "Poor voice cloning quality"

**Problem:**
Cloned voice doesn't sound like the reference audio.

**Solutions:**

**1. Improve Reference Audio**
- Use 6-10 seconds of clean audio
- Ensure minimal background noise
- Use consistent audio quality
- Single speaker only

**2. Upload Multiple Samples**
- Upload 2-3 different recordings of same voice
- App will combine them automatically
- More samples = better quality

**3. Check Audio Format**
- Use WAV or high-quality MP3
- Avoid heavily compressed audio
- 16kHz or higher sample rate recommended

**4. Use English Only**
- Voice cloning only works with English model
- Indonesian model doesn't support cloning

---

### ❌ Error: "Streamlit connection error"

**Problem:**
Browser can't connect to Streamlit server.

**Solutions:**

**1. Check if Server is Running**
```bash
# Should see: "You can now view your Streamlit app in your browser"
# If not, restart the app
streamlit run streamlit_tts_app.py
```

**2. Try Different Port**
```bash
streamlit run streamlit_tts_app.py --server.port 8080
```

**3. Check Firewall**
- Ensure port 8501 (or custom port) is not blocked
- Add exception for Streamlit if needed

**4. Use Specific Address**
```bash
# For local access only
streamlit run streamlit_tts_app.py --server.address localhost

# For network access
streamlit run streamlit_tts_app.py --server.address 0.0.0.0
```

---

### ❌ Error: "Torch not available" or "No module named 'torch'"

**Problem:**
PyTorch is not installed.

**Solutions:**

**1. Install from Requirements**
```bash
pip install -r requirements.txt
```

**2. Install PyTorch Manually**

**For CPU only:**
```bash
pip install torch torchaudio
```

**For CUDA 11.8:**
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**For CUDA 12.1:**
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**3. Verify Installation**
```bash
python3 -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

---

### ❌ Error: "Processing too slow"

**Problem:**
Speech synthesis takes too long.

**Solutions:**

**1. Enable GPU**
- Check "Use GPU Acceleration" in sidebar
- Verify GPU is detected

**2. Verify GPU is Working**
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device: {torch.cuda.get_device_name(0)}")
```

**3. Use Smaller Text Chunks**
- Process shorter texts
- Enable sentence splitting

**4. Upgrade Hardware**
- CPU: 8+ cores recommended
- GPU: RTX 2060 or better recommended
- RAM: 16GB recommended

---

### ❌ Error: "Audio file upload failed"

**Problem:**
Reference audio file won't upload.

**Solutions:**

**1. Check File Format**
- Supported: WAV, MP3, FLAC, OGG
- Not supported: AAC, M4A, WMA

**2. Convert Unsupported Formats**
```bash
# Using ffmpeg
ffmpeg -i input.m4a output.wav
```

**3. Check File Size**
- Files should be reasonable size (<100MB)
- Very large files may timeout

**4. Verify Audio Integrity**
- Ensure file is not corrupted
- Try playing in media player first

---

### ❌ Error: "Language not supported"

**Problem:**
Text language doesn't match selected model.

**Solutions:**

**1. For English Text**
- Select "English (XTTS v2)" in sidebar
- XTTS v2 supports 17 languages
- But voice cloning works best with English

**2. For Indonesian Text**
- Select "Indonesian (Fairseq VITS)" in sidebar
- No voice cloning available for Indonesian

**3. For Other Languages**
- XTTS v2 supports: en, es, fr, de, it, pt, pl, tr, ru, nl, cs, ar, zh, hu, ko, ja, hi
- Use XTTS v2 model for these languages

---

## Environment-Specific Issues

### macOS

**Apple Silicon (M1/M2/M3)**
```bash
# Use MPS acceleration instead of CUDA
# PyTorch should automatically detect and use MPS
# Or force CPU mode if issues occur
```

**Permission Issues**
```bash
# If file permissions error
chmod +x streamlit_tts_app.py
```

### Windows

**Long Path Names**
```bash
# Enable long paths in Windows
# Run as Administrator:
reg add HKLM\SYSTEM\CurrentControlSet\Control\FileSystem /v LongPathsEnabled /t REG_DWORD /d 1
```

**Microsoft Visual C++ Error**
```bash
# Install Visual C++ Redistributable
# Download from: https://aka.ms/vs/17/release/vc_redist.x64.exe
```

### Linux

**CUDA Issues**
```bash
# Verify CUDA installation
nvidia-smi

# Check CUDA version
nvcc --version

# Install correct PyTorch for your CUDA version
```

**Audio Backend Issues**
```bash
# Install audio libraries
sudo apt-get install libsndfile1 ffmpeg
```

---

## Diagnostic Commands

Run these to gather diagnostic information:

```bash
# Check Python version
python3 --version

# Check installed packages
pip list | grep -E "numpy|torch|streamlit|TTS|numba|librosa"

# Check CUDA
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Check NumPy version
python3 -c "import numpy; print(f'NumPy: {numpy.__version__}')"

# Check TTS installation
python3 -c "from TTS.api import TTS; print('TTS imported successfully')"

# Check Streamlit
streamlit --version
```

---

## Getting More Help

If your issue isn't listed here:

1. **Check Full Documentation**: `STREAMLIT_TTS_GUIDE.md`
2. **Coqui TTS Issues**: https://github.com/coqui-ai/TTS/issues
3. **Coqui TTS Docs**: https://tts.readthedocs.io/
4. **Streamlit Docs**: https://docs.streamlit.io/

When reporting issues, include:
- Error message (full traceback)
- Python version
- Operating system
- Output of diagnostic commands above
- Steps to reproduce

---

**Last Updated**: 2025-11-10
