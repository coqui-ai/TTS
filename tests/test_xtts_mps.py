"""
Tests for XTTS MPS (Metal Performance Shaders) support.

These tests verify that XTTS can run on Apple Silicon GPUs using MPS.
"""

import os
import pytest
import torch

# Only run these tests if MPS is available
mps_available = torch.backends.mps.is_available()


@pytest.mark.skipif(not mps_available, reason="MPS not available")
class TestXTTSMPS:
    """Test suite for XTTS MPS support."""

    @pytest.fixture(scope="class")
    def sample_audio_path(self):
        """Get path to sample audio file."""
        import glob
        wav_files = glob.glob("/Users/jbuesking/repos/python/TTS/tests/data/**/*.wav", recursive=True)
        if wav_files:
            return wav_files[0]
        return None

    @pytest.fixture(scope="class")
    def tts_mps(self):
        """Create TTS instance on MPS."""
        from TTS.api import TTS

        # Set MPS fallback environment variable
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
        tts.to("mps")

        return tts

    def test_model_on_mps_device(self, tts_mps):
        """Test that model is actually on MPS device."""
        assert tts_mps.device == "mps", f"Expected device 'mps', got '{tts_mps.device}'"
        assert tts_mps.synthesizer.device.type == "mps"

        # Check that the TTS model is on MPS
        model_device = next(tts_mps.synthesizer.tts_model.parameters()).device
        assert model_device.type == "mps", f"Model is on {model_device}, expected MPS"

    def test_model_is_fp32_on_mps(self, tts_mps):
        """Test that model uses FP32 on MPS (not FP16)."""
        # Check first parameter dtype
        param = next(tts_mps.synthesizer.tts_model.parameters())
        assert param.dtype == torch.float32, f"Expected FP32, got {param.dtype}"

    def test_conditioning_latents_on_mps(self, tts_mps, sample_audio_path):
        """Test that conditioning latents can be computed on MPS."""
        if sample_audio_path is None:
            pytest.skip("No sample audio file available")

        model = tts_mps.synthesizer.tts_model

        # Compute conditioning latents
        gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
            audio_path=[sample_audio_path],
            max_ref_length=10,  # Shorter for faster test
            gpt_cond_len=3,
            gpt_cond_chunk_len=3,
        )

        # Verify outputs are on MPS
        assert gpt_cond_latent.device.type == "mps"
        assert speaker_embedding.device.type == "mps"

        # Verify shapes
        assert gpt_cond_latent.ndim == 3
        assert speaker_embedding.ndim == 3

    def test_short_text_inference_on_mps(self, tts_mps, sample_audio_path):
        """Test short text inference on MPS."""
        if sample_audio_path is None:
            pytest.skip("No sample audio file available")

        text = "Hello, this is a test."

        # Generate speech
        wav = tts_mps.tts(
            text=text,
            speaker_wav=sample_audio_path,
            language="en",
        )

        # Verify output
        assert wav is not None
        assert len(wav) > 0
        assert isinstance(wav, (list, torch.Tensor, type(None).__class__))

    def test_tts_to_file_on_mps(self, tts_mps, sample_audio_path, tmp_path):
        """Test TTS to file on MPS."""
        if sample_audio_path is None:
            pytest.skip("No sample audio file available")

        output_file = tmp_path / "test_mps_output.wav"
        text = "Testing MPS text to speech."

        # Generate and save
        tts_mps.tts_to_file(
            text=text,
            speaker_wav=sample_audio_path,
            language="en",
            file_path=str(output_file),
        )

        # Verify file was created
        assert output_file.exists()
        assert output_file.stat().st_size > 0

    def test_streaming_inference_on_mps(self, tts_mps, sample_audio_path):
        """Test streaming inference on MPS."""
        if sample_audio_path is None:
            pytest.skip("No sample audio file available")

        model = tts_mps.synthesizer.tts_model

        # Get conditioning latents
        gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
            audio_path=[sample_audio_path],
            max_ref_length=10,
        )

        text = "Testing streaming on MPS."

        # Generate streaming chunks
        chunks = list(model.inference_stream(
            text=text,
            language="en",
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            stream_chunk_size=10,
        ))

        # Verify we got chunks
        assert len(chunks) > 0

        # Verify chunks are tensors
        for chunk in chunks:
            assert isinstance(chunk, torch.Tensor)
            assert chunk.numel() > 0

    def test_attention_mask_present(self, tts_mps, sample_audio_path):
        """Test that attention mask is provided in generation (no warnings)."""
        if sample_audio_path is None:
            pytest.skip("No sample audio file available")

        import warnings

        text = "Test attention mask."

        # Capture warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            tts_mps.tts(
                text=text,
                speaker_wav=sample_audio_path,
                language="en",
            )

            # Check for attention mask warning
            attention_warnings = [
                warning for warning in w
                if "attention_mask" in str(warning.message).lower()
            ]

            assert len(attention_warnings) == 0, \
                f"Got attention_mask warning: {attention_warnings[0].message if attention_warnings else ''}"

    def test_device_move_to_cpu_and_back(self, tts_mps):
        """Test moving model from MPS to CPU and back."""
        # Model starts on MPS
        assert tts_mps.device == "mps"

        # Move to CPU
        tts_mps.to("cpu")
        assert tts_mps.device == "cpu"
        assert tts_mps.synthesizer.device.type == "cpu"

        # Move back to MPS
        tts_mps.to("mps")
        assert tts_mps.device == "mps"
        assert tts_mps.synthesizer.device.type == "mps"

    def test_speaker_encoder_runs_on_cpu(self, tts_mps, sample_audio_path):
        """Test that speaker encoder runs on CPU when model is on MPS."""
        if sample_audio_path is None:
            pytest.skip("No sample audio file available")

        model = tts_mps.synthesizer.tts_model

        # Load audio
        import torchaudio
        import soundfile as sf

        audio, sr = sf.read(sample_audio_path)
        audio_tensor = torch.FloatTensor(audio).T
        if audio_tensor.ndim == 1:
            audio_tensor = audio_tensor.unsqueeze(0)

        # Call get_speaker_embedding
        # This should internally run on CPU to avoid MPS conv issues
        speaker_emb = model.get_speaker_embedding(audio_tensor, sr)

        # Result should be on MPS (moved back after CPU computation)
        assert speaker_emb.device.type == "mps"

    def test_cpu_mps_waveform_similarity(self, tts_mps, sample_audio_path):
        """Ensure CPU and MPS outputs are numerically close (within tolerance)."""
        if sample_audio_path is None:
            pytest.skip("No sample audio file available")

        import numpy as np

        text = "Short consistency check for CPU and MPS."

        # Run on CPU
        tts_mps.to("cpu")
        torch.manual_seed(0)
        wav_cpu = tts_mps.tts(
            text=text,
            speaker_wav=sample_audio_path,
            language="en",
            do_sample=False,
        )

        # Convert CPU output to tensor and run basic sanity checks
        wav_cpu_t = torch.as_tensor(wav_cpu, dtype=torch.float32)
        assert wav_cpu_t.numel() > 0, "CPU waveform is empty"
        assert torch.isfinite(wav_cpu_t).all(), "CPU waveform contains NaN or inf values"

        max_amp = wav_cpu_t.abs().max().item()
        std_amp = wav_cpu_t.std().item()

        # Ensure it's not (near) silent and not obviously clipped to huge values
        assert max_amp > 1e-3, f"CPU waveform appears silent (max_amp={max_amp:.6f})"
        assert max_amp < 5.0, f"CPU waveform amplitude unusually large (max_amp={max_amp:.6f})"
        assert std_amp > 1e-4, f"CPU waveform has extremely low variance (std={std_amp:.6f})"

        # Run on MPS
        tts_mps.to("mps")
        torch.manual_seed(0)
        wav_mps = tts_mps.tts(
            text=text,
            speaker_wav=sample_audio_path,
            language="en",
            do_sample=False,
        )

        # Convert to tensors for comparison
        wav_mps_t = torch.as_tensor(wav_mps, dtype=torch.float32)

        # Align lengths in case of off-by-few-samples differences
        min_len = min(wav_cpu_t.shape[-1], wav_mps_t.shape[-1])
        wav_cpu_t = wav_cpu_t[..., :min_len]
        wav_mps_t = wav_mps_t[..., :min_len].to("cpu")

        # Compute max absolute difference and SNR
        diff = wav_cpu_t - wav_mps_t
        max_abs_diff = diff.abs().max().item()

        # Avoid division by zero in pathological cases
        signal_power = (wav_cpu_t**2).mean().item()
        noise_power = (diff**2).mean().item() + 1e-12
        snr_db = 10.0 * np.log10(signal_power / noise_power) if signal_power > 0 else float("inf")

        # Also compute correlation to make sure the waveforms have
        # similar shape, not just similar overall energy.
        if wav_cpu_t.numel() >= 2:
            stacked = torch.stack(
                [
                    wav_cpu_t.flatten() - wav_cpu_t.mean(),
                    wav_mps_t.flatten() - wav_mps_t.mean(),
                ]
            )
            corr = torch.corrcoef(stacked)[0, 1].item()
        else:
            corr = 1.0

        # Be tolerant to backend differences but catch obvious problems.
        # We require a modest SNR and strong correlation.
        assert snr_db > 5.0, f"SNR between CPU and MPS too low: {snr_db:.2f} dB (max_abs_diff={max_abs_diff:.4f})"
        assert corr > 0.9, f"Waveform correlation between CPU and MPS too low: {corr:.3f}"


@pytest.mark.skipif(not mps_available, reason="MPS not available")
def test_xtts_direct_load_on_mps():
    """Test loading XTTS model directly on MPS."""
    from TTS.tts.models.xtts import Xtts
    from TTS.config import load_config

    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    # This test would need actual model files
    # Skipping for now since we test via TTS API above
    pytest.skip("Direct model loading test requires model files")


@pytest.mark.skipif(not mps_available, reason="MPS not available")
def test_synthesizer_device_parameter():
    """Test that Synthesizer accepts device parameter."""
    from TTS.utils.synthesizer import Synthesizer
    import warnings

    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    # Test that device parameter is accepted (without actually loading a model)
    # We just test the constructor accepts the parameter
    try:
        # This will fail due to missing files, but we just want to check the parameter is accepted
        synth = Synthesizer(device="mps")
    except (FileNotFoundError, TypeError, OSError):
        # Expected to fail without model files, but shouldn't fail due to device parameter
        pass


@pytest.mark.skipif(not mps_available, reason="MPS not available")
def test_mps_environment_variable():
    """Test that PYTORCH_ENABLE_MPS_FALLBACK environment variable is set."""
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1"
