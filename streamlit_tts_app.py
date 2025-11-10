#!/usr/bin/env python3
"""
Coqui TTS Web Interface - Streamlit Application

A web interface for text-to-speech synthesis with voice cloning capabilities.
Supports English (with voice cloning via XTTS v2) and Indonesian (via Fairseq VITS).
"""

import streamlit as st
import torch
import tempfile
import os
from pathlib import Path
import time
from TTS.api import TTS
import soundfile as sf
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Coqui TTS - Voice Cloning & Synthesis",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f0f2f6;
        margin: 1rem 0;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    </style>
""", unsafe_allow_html=True)

# Model configurations
MODEL_CONFIGS = {
    "English (XTTS v2)": {
        "model_name": "tts_models/multilingual/multi-dataset/xtts_v2",
        "language_code": "en",
        "supports_cloning": True,
        "description": "High-quality voice cloning with XTTS v2"
    },
    "Indonesian (Fairseq VITS)": {
        "model_name": "tts_models/ind/fairseq/vits",
        "language_code": "ind",
        "supports_cloning": False,
        "description": "Standard Indonesian TTS (no voice cloning)"
    }
}

@st.cache_resource
def load_tts_model(model_name: str, use_gpu: bool = True):
    """
    Load and cache TTS model.

    Args:
        model_name: Name of the TTS model to load
        use_gpu: Whether to use GPU acceleration

    Returns:
        TTS: Loaded TTS model instance
    """
    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    with st.spinner(f"Loading model {model_name}... This may take a few minutes on first run."):
        tts = TTS(model_name, progress_bar=False, gpu=use_gpu)
        if use_gpu and torch.cuda.is_available():
            tts = tts.to(device)
    return tts

def save_uploaded_file(uploaded_file) -> str:
    """
    Save uploaded file to temporary location.

    Args:
        uploaded_file: Streamlit UploadedFile object

    Returns:
        str: Path to saved temporary file
    """
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            return tmp_file.name
    except Exception as e:
        st.error(f"Error saving uploaded file: {e}")
        return None

def combine_audio_files(audio_paths: list) -> str:
    """
    Combine multiple audio files into a single reference file.

    Args:
        audio_paths: List of paths to audio files

    Returns:
        str: Path to combined audio file
    """
    if len(audio_paths) == 1:
        return audio_paths[0]

    try:
        combined_audio = []
        sample_rate = None

        for audio_path in audio_paths:
            audio_data, sr = sf.read(audio_path)
            if sample_rate is None:
                sample_rate = sr
            elif sr != sample_rate:
                # Resample if needed (simple approach - just warning for now)
                st.warning(f"Audio files have different sample rates. Using {sample_rate} Hz.")
            combined_audio.append(audio_data)

        # Concatenate audio
        combined = np.concatenate(combined_audio)

        # Save combined audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            sf.write(tmp_file.name, combined, sample_rate)
            return tmp_file.name

    except Exception as e:
        st.error(f"Error combining audio files: {e}")
        return None

def synthesize_speech(tts_model, text: str, language: str, speaker_wav: str = None,
                     split_sentences: bool = True) -> tuple:
    """
    Synthesize speech from text.

    Args:
        tts_model: Loaded TTS model
        text: Input text to synthesize
        language: Language code
        speaker_wav: Path to speaker reference audio (for voice cloning)
        split_sentences: Whether to split text into sentences

    Returns:
        tuple: (audio_data, sample_rate, output_path)
    """
    try:
        # Create temporary output file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            output_path = tmp_file.name

        # Synthesize based on whether voice cloning is used
        if speaker_wav and language == "en":
            # XTTS v2 with voice cloning
            tts_model.tts_to_file(
                text=text,
                speaker_wav=speaker_wav,
                language=language,
                file_path=output_path,
                split_sentences=split_sentences
            )
        else:
            # Standard synthesis (Fairseq for Indonesian)
            tts_model.tts_to_file(
                text=text,
                file_path=output_path,
                split_sentences=split_sentences
            )

        # Load the generated audio
        audio_data, sample_rate = sf.read(output_path)

        return audio_data, sample_rate, output_path

    except Exception as e:
        st.error(f"Error during synthesis: {e}")
        return None, None, None

def main():
    """Main application function"""

    # Header
    st.markdown('<p class="main-header">🎙️ Coqui TTS - Voice Cloning & Synthesis</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Clone voices and synthesize speech in English and Indonesian</p>', unsafe_allow_html=True)

    # Sidebar - Settings and Configuration
    with st.sidebar:
        st.header("⚙️ Settings")

        # Language/Model Selection
        selected_language = st.selectbox(
            "Select Language & Model",
            options=list(MODEL_CONFIGS.keys()),
            help="Choose the language and TTS model to use"
        )

        model_config = MODEL_CONFIGS[selected_language]

        # Display model info
        st.info(f"**Model:** {model_config['model_name']}\n\n**Features:** {model_config['description']}")

        # GPU settings
        use_gpu = st.checkbox(
            "Use GPU Acceleration",
            value=torch.cuda.is_available(),
            disabled=not torch.cuda.is_available(),
            help="Enable GPU for faster synthesis (if available)"
        )

        if torch.cuda.is_available():
            st.success("✓ GPU Available")
        else:
            st.warning("CPU Mode (GPU not available)")

        # Advanced settings
        with st.expander("🔧 Advanced Settings"):
            split_sentences = st.checkbox(
                "Split into Sentences",
                value=True,
                help="Split long text into sentences for better quality and memory efficiency"
            )

            show_waveform = st.checkbox(
                "Show Waveform",
                value=False,
                help="Display audio waveform visualization"
            )

        # Info section
        st.markdown("---")
        st.markdown("### 📚 About")
        st.markdown("""
        This interface uses Coqui TTS:
        - **English**: XTTS v2 with voice cloning
        - **Indonesian**: Fairseq VITS (standard)

        Supports long-form text synthesis with automatic sentence splitting.
        """)

    # Main content area
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("📝 Text Input")

        # Text input method selection
        input_method = st.radio(
            "Input Method",
            ["Type Text", "Upload File"],
            horizontal=True
        )

        input_text = ""

        if input_method == "Type Text":
            input_text = st.text_area(
                "Enter text to synthesize",
                height=300,
                placeholder="Enter your text here...\n\nYou can enter long-form text, and it will be automatically split into sentences for optimal synthesis.",
                help="Enter the text you want to convert to speech"
            )
        else:
            uploaded_text_file = st.file_uploader(
                "Upload Text File",
                type=["txt"],
                help="Upload a .txt file containing the text to synthesize"
            )

            if uploaded_text_file:
                input_text = uploaded_text_file.read().decode("utf-8")
                st.text_area("File Contents", value=input_text, height=300, disabled=True)

        # Display text statistics
        if input_text:
            word_count = len(input_text.split())
            char_count = len(input_text)
            st.caption(f"📊 Characters: {char_count} | Words: {word_count}")

    with col2:
        st.header("🎤 Voice Cloning")

        if model_config["supports_cloning"]:
            st.markdown("""
            Upload one or more audio recordings (3+ seconds each) to clone a voice.
            The model will learn the voice characteristics and apply them to the generated speech.
            """)

            uploaded_audio_files = st.file_uploader(
                "Upload Reference Audio File(s)",
                type=["wav", "mp3", "flac", "ogg"],
                accept_multiple_files=True,
                help="Upload audio files to clone the voice. Multiple files will be combined."
            )

            if uploaded_audio_files:
                st.success(f"✓ {len(uploaded_audio_files)} audio file(s) uploaded")

                # Preview uploaded audio files
                with st.expander("🔊 Preview Reference Audio"):
                    for idx, audio_file in enumerate(uploaded_audio_files, 1):
                        st.write(f"**File {idx}:** {audio_file.name}")
                        st.audio(audio_file, format=f'audio/{Path(audio_file.name).suffix[1:]}')
        else:
            st.info("""
            ℹ️ **Voice cloning is not available for Indonesian.**

            The Fairseq VITS model uses a standard voice for synthesis.

            To enable voice cloning for Indonesian, consider using the voice conversion feature (coming soon) or switching to English.
            """)
            uploaded_audio_files = None

    # Generation section
    st.markdown("---")
    st.header("🚀 Generate Speech")

    col_gen1, col_gen2, col_gen3 = st.columns([2, 1, 1])

    with col_gen1:
        generate_button = st.button(
            "🎯 Generate Speech",
            type="primary",
            use_container_width=True,
            disabled=not input_text
        )

    with col_gen2:
        if input_text:
            estimated_duration = len(input_text.split()) / 150 * 60  # Rough estimate: 150 words/min
            st.metric("Est. Audio Duration", f"{estimated_duration:.1f}s")

    with col_gen3:
        if input_text:
            st.metric("Processing Time", "~5-30s")

    # Generation process
    if generate_button:
        if not input_text.strip():
            st.error("❌ Please enter some text to synthesize.")
            return

        # Load model
        try:
            tts_model = load_tts_model(model_config["model_name"], use_gpu)
        except Exception as e:
            st.error(f"❌ Error loading model: {e}")
            return

        # Process reference audio if available
        speaker_wav_path = None
        if uploaded_audio_files and model_config["supports_cloning"]:
            with st.spinner("Processing reference audio..."):
                # Save uploaded files
                temp_audio_paths = []
                for audio_file in uploaded_audio_files:
                    temp_path = save_uploaded_file(audio_file)
                    if temp_path:
                        temp_audio_paths.append(temp_path)

                if temp_audio_paths:
                    # Combine if multiple files
                    speaker_wav_path = combine_audio_files(temp_audio_paths)

                    if not speaker_wav_path:
                        st.error("❌ Error processing reference audio.")
                        return

        # Synthesize speech
        with st.spinner("🎵 Synthesizing speech... This may take a moment for long texts."):
            start_time = time.time()

            audio_data, sample_rate, output_path = synthesize_speech(
                tts_model,
                input_text,
                model_config["language_code"],
                speaker_wav_path,
                split_sentences
            )

            end_time = time.time()
            processing_time = end_time - start_time

        # Display results
        if audio_data is not None:
            st.markdown('<div class="success-box">✅ <strong>Speech generated successfully!</strong></div>', unsafe_allow_html=True)

            # Results section
            st.header("🎧 Generated Audio")

            col_result1, col_result2 = st.columns([2, 1])

            with col_result1:
                # Audio player
                st.audio(output_path, format="audio/wav")

                # Download button
                with open(output_path, "rb") as audio_file:
                    st.download_button(
                        label="📥 Download Audio (WAV)",
                        data=audio_file,
                        file_name="synthesized_speech.wav",
                        mime="audio/wav",
                        use_container_width=True
                    )

            with col_result2:
                # Statistics
                duration = len(audio_data) / sample_rate
                st.metric("Audio Duration", f"{duration:.2f}s")
                st.metric("Processing Time", f"{processing_time:.2f}s")
                st.metric("Sample Rate", f"{sample_rate} Hz")

            # Waveform visualization (optional)
            if show_waveform:
                st.subheader("📊 Waveform")
                try:
                    import matplotlib.pyplot as plt

                    fig, ax = plt.subplots(figsize=(12, 3))
                    time_axis = np.linspace(0, duration, len(audio_data))
                    ax.plot(time_axis, audio_data, linewidth=0.5)
                    ax.set_xlabel("Time (s)")
                    ax.set_ylabel("Amplitude")
                    ax.set_title("Audio Waveform")
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                except ImportError:
                    st.info("Install matplotlib to view waveform visualization")

        # Cleanup temporary files
        try:
            if speaker_wav_path and os.path.exists(speaker_wav_path):
                os.unlink(speaker_wav_path)
            if uploaded_audio_files:
                for temp_path in temp_audio_paths:
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
        except Exception as e:
            pass  # Silent cleanup failure

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>Built with <a href="https://github.com/coqui-ai/TTS" target="_blank">Coqui TTS</a> |
        Powered by <a href="https://streamlit.io" target="_blank">Streamlit</a></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
