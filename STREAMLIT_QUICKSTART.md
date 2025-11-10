# Coqui TTS Web Interface - Quick Start Guide

A simple web interface for voice cloning and text-to-speech in English and Indonesian.

## Quick Installation

### Step 1: Install Coqui TTS (Required First!)

You **must** install the base TTS package before running the Streamlit app:

```bash
# Install TTS with all dependencies
pip install -e .[all]
```

This installs all the core TTS dependencies including coqpit, torch, librosa, etc.

### Step 2: Install Streamlit Requirements

```bash
# Install Streamlit and optional dependencies
pip install -r requirements.streamlit.txt
```

### Step 3: Run the App

```bash
streamlit run streamlit_tts_app.py
```

The app will open automatically in your browser at `http://localhost:8501`

---

### ⚠️ Common Installation Issues

**Error: "No module named 'coqpit'" or similar import errors**
```bash
# You need to install TTS first!
pip install -e .[all]
```

**Error: "Numba needs NumPy 2.2 or less"**
```bash
pip install "numpy>=1.24.3,<2.0"
```

**Error: "Weights only load failed" (PyTorch 2.6)**
- This has been fixed in the code! Pull the latest version.
- Alternatively, downgrade: `pip install "torch>=2.1,<2.6"`

**Error: "'GPT2InferenceModel' object has no attribute 'generate'" (Transformers 4.50+)**
- This has been fixed in the code! Pull the latest version.
- Alternatively, downgrade: `pip install "transformers>=4.33.0,<4.50"`

**See `TROUBLESHOOTING.md` for more solutions.**

## 5-Minute Tutorial

### For English Voice Cloning

1. **Select Language**: Choose "English (XTTS v2)" in the sidebar
2. **Upload Voice**: Upload a 3-10 second audio file of the voice you want to clone
3. **Enter Text**: Type or paste your text in the text area
4. **Generate**: Click "Generate Speech"
5. **Download**: Play the audio and download if satisfied

### For Indonesian Text-to-Speech

1. **Select Language**: Choose "Indonesian (Fairseq VITS)" in the sidebar
2. **Enter Text**: Type or paste your Indonesian text
3. **Generate**: Click "Generate Speech"
4. **Download**: Play the audio and download if satisfied

## Features

✅ **Voice Cloning** (English only) - Clone any voice with a short audio sample
✅ **Long-Form Text** - Process articles, documents, or books automatically
✅ **Multiple Audio Files** - Upload multiple voice samples for better quality
✅ **GPU Acceleration** - Automatic GPU detection for 5-10x faster synthesis
✅ **Preview & Download** - Listen and download generated audio

## Supported Languages

- **English**: With voice cloning (XTTS v2)
- **Indonesian**: Standard synthesis (Fairseq VITS)

## Audio Requirements for Voice Cloning

- **Format**: WAV, MP3, FLAC, or OGG
- **Duration**: 3-10 seconds (6 seconds ideal)
- **Quality**: Clear speech, minimal background noise
- **Speaker**: Single speaker only
- **Tip**: Multiple samples improve quality

## System Requirements

**Minimum:**
- 8GB RAM
- 5GB free disk space

**Recommended:**
- 16GB RAM
- NVIDIA GPU with 6GB+ VRAM
- 10GB free disk space (SSD)

## First Run

On first run, the app will:
1. Download the selected TTS model (~2GB for XTTS v2)
2. Cache the model for future use
3. This is a one-time process

## Troubleshooting

**App won't start?**
```bash
pip install streamlit --upgrade
```

**Out of memory?**
- Enable "Split into Sentences" in Advanced Settings
- Disable GPU (use CPU mode)
- Process shorter texts

**Poor voice quality?**
- Use longer reference audio (6+ seconds)
- Upload multiple samples
- Ensure clean, clear audio
- Check for single speaker

## Examples

### Example 1: Clone Your Voice

```text
1. Record yourself reading:
   "The quick brown fox jumps over the lazy dog.
    Pack my box with five dozen liquor jugs."

2. Upload the recording
3. Enter any text you want in your voice
4. Generate and download!
```

### Example 2: Long Article

```text
1. Copy an entire news article or blog post
2. Paste into the text area
3. Enable "Split into Sentences"
4. Generate (may take 30-60 seconds)
5. Download the complete audio
```

## Next Steps

- Read the full guide: `STREAMLIT_TTS_GUIDE.md`
- Explore advanced settings in the sidebar
- Try different voices and languages
- Process long-form content

## Support

- **Full Documentation**: See `STREAMLIT_TTS_GUIDE.md`
- **Coqui TTS Docs**: https://tts.readthedocs.io/
- **Issues**: https://github.com/coqui-ai/TTS/issues

---

**Have fun creating synthetic speech!** 🎙️
