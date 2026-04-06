"""
Tests for TTS.minimax_tts.MiniMaxTTS.

Unit tests mock all HTTP calls so they run without a real API key.
Integration tests are skipped automatically when MINIMAX_API_KEY is not set.
"""

import io
import json
import os
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from TTS.minimax_tts import (
    MINIMAX_MODELS,
    MINIMAX_VOICES,
    MiniMaxTTS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sse_response(hex_chunks):
    """Build a fake SSE byte-stream from a list of hex strings."""
    lines = []
    for i, hex_chunk in enumerate(hex_chunks):
        event = {"data": {"audio": hex_chunk, "status": 1 if i < len(hex_chunks) - 1 else 2}}
        lines.append(b"data: " + json.dumps(event).encode() + b"\n\n")
    lines.append(b"data: [DONE]\n\n")
    return b"".join(lines)


def _make_mp3_bytes():
    """Return a minimal valid MP3 header (ID3 tag) for mocking purposes."""
    # Tiny ID3v2 header — enough to let torchaudio attempt to decode
    return bytes.fromhex("494433")  # "ID3" magic


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


class TestMiniMaxTTSInit(unittest.TestCase):
    def test_valid_init(self):
        tts = MiniMaxTTS(api_key="test-key")
        self.assertIsInstance(tts, MiniMaxTTS)
        self.assertEqual(tts.model, "speech-2.8-hd")
        self.assertEqual(tts.base_url, "https://api.minimax.io")

    def test_custom_model(self):
        tts = MiniMaxTTS(model="speech-2.8-turbo", api_key="test-key")
        self.assertEqual(tts.model, "speech-2.8-turbo")

    def test_invalid_model(self):
        with self.assertRaises(ValueError):
            MiniMaxTTS(model="non-existent-model", api_key="test-key")

    def test_missing_api_key(self):
        env = {k: v for k, v in os.environ.items() if k != "MINIMAX_API_KEY"}
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaises(ValueError):
                MiniMaxTTS()

    def test_api_key_from_env(self):
        with patch.dict(os.environ, {"MINIMAX_API_KEY": "env-key"}):
            tts = MiniMaxTTS()
            self.assertEqual(tts._api_key, "env-key")

    def test_trailing_slash_stripped(self):
        tts = MiniMaxTTS(api_key="k", base_url="https://api.minimax.io/")
        self.assertEqual(tts.base_url, "https://api.minimax.io")

    def test_voices_property(self):
        tts = MiniMaxTTS(api_key="k")
        self.assertEqual(tts.voices, MINIMAX_VOICES)


class TestMiniMaxTTSCallApi(unittest.TestCase):
    """Tests for _call_api — the HTTP request is mocked."""

    def _mock_response(self, hex_chunks):
        sse_bytes = _make_sse_response(hex_chunks)
        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        # Simulate streaming in 64-byte blocks
        blocks = [sse_bytes[i : i + 64] for i in range(0, len(sse_bytes), 64)]
        blocks.append(b"")  # EOF
        mock_resp.read.side_effect = blocks
        return mock_resp

    def test_call_api_collects_hex_chunks(self):
        # Two hex chunks: "deadbeef" and "cafebabe"
        hex_chunks = ["deadbeef", "cafebabe"]
        expected = bytes.fromhex("deadbeef") + bytes.fromhex("cafebabe")

        tts = MiniMaxTTS(api_key="test-key")
        mock_resp = self._mock_response(hex_chunks)
        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = tts._call_api("Hello", "English_Graceful_Lady", 1.0, 1.0, 0)

        self.assertEqual(result, expected)

    def test_call_api_raises_on_empty_audio(self):
        tts = MiniMaxTTS(api_key="test-key")
        mock_resp = self._mock_response([])  # no audio chunks
        with patch("urllib.request.urlopen", return_value=mock_resp):
            with self.assertRaises(RuntimeError):
                tts._call_api("Hello", "English_Graceful_Lady", 1.0, 1.0, 0)

    def test_call_api_request_url(self):
        """The request must target the correct endpoint."""
        captured = {}

        def fake_urlopen(req):
            captured["url"] = req.full_url
            captured["method"] = req.get_method()
            raise RuntimeError("stop")  # prevent further execution

        tts = MiniMaxTTS(api_key="test-key")
        with self.assertRaises(RuntimeError):
            with patch("urllib.request.urlopen", side_effect=fake_urlopen):
                tts._call_api("Hello", "English_Graceful_Lady", 1.0, 1.0, 0)

        self.assertIn("/v1/t2a_v2", captured["url"])

    def test_call_api_authorization_header(self):
        """Authorization header must carry the Bearer token."""
        captured_headers = {}

        def fake_urlopen(req):
            captured_headers.update(dict(req.headers))
            raise RuntimeError("stop")

        tts = MiniMaxTTS(api_key="my-secret-key")
        with self.assertRaises(RuntimeError):
            with patch("urllib.request.urlopen", side_effect=fake_urlopen):
                tts._call_api("Hello", "English_Graceful_Lady", 1.0, 1.0, 0)

        # Header keys are capitalised by urllib
        auth = captured_headers.get("Authorization", "")
        self.assertEqual(auth, "Bearer my-secret-key")


class TestMiniMaxTTSMp3ToNumpy(unittest.TestCase):
    """Tests for _mp3_to_numpy using torchaudio mock."""

    def test_converts_mp3_to_float32(self):
        import torch

        expected = np.array([0.1, -0.2, 0.3], dtype=np.float32)
        mock_tensor = torch.tensor(expected).unsqueeze(0)  # shape [1, 3]

        with patch("torchaudio.load", return_value=(mock_tensor, 32000)):
            result = MiniMaxTTS._mp3_to_numpy(b"fake-mp3")

        np.testing.assert_array_almost_equal(result, expected)
        self.assertEqual(result.dtype, np.float32)

    def test_stereo_converted_to_mono(self):
        import torch

        ch1 = torch.tensor([1.0, 2.0])
        ch2 = torch.tensor([3.0, 4.0])
        stereo = torch.stack([ch1, ch2])  # shape [2, 2]
        expected_mono = np.array([2.0, 3.0], dtype=np.float32)

        with patch("torchaudio.load", return_value=(stereo, 32000)):
            result = MiniMaxTTS._mp3_to_numpy(b"fake-mp3")

        np.testing.assert_array_almost_equal(result, expected_mono)


class TestMiniMaxTTSTts(unittest.TestCase):
    """Tests for the public tts() method."""

    def _make_tts_with_mock(self, wav_array):
        tts = MiniMaxTTS(api_key="test-key")
        tts._call_api = MagicMock(return_value=b"mp3-bytes")
        tts._mp3_to_numpy = MagicMock(return_value=wav_array)
        return tts

    def test_tts_returns_numpy_array(self):
        fake_wav = np.zeros(1000, dtype=np.float32)
        tts = self._make_tts_with_mock(fake_wav)
        result = tts.tts("Hello world")
        np.testing.assert_array_equal(result, fake_wav)

    def test_tts_passes_voice_to_api(self):
        tts = MiniMaxTTS(api_key="k")
        tts._call_api = MagicMock(return_value=b"mp3")
        tts._mp3_to_numpy = MagicMock(return_value=np.zeros(10, dtype=np.float32))
        tts.tts("Hi", voice="English_Lucky_Robot")
        tts._call_api.assert_called_once_with(
            text="Hi",
            voice="English_Lucky_Robot",
            speed=1.0,
            volume=1.0,
            pitch=0,
        )

    def test_tts_to_file_saves_wav(self):
        import tempfile

        fake_wav = np.zeros(1000, dtype=np.float32)
        tts = MiniMaxTTS(api_key="k")
        tts.tts = MagicMock(return_value=fake_wav)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            out_path = f.name

        try:
            with patch("TTS.utils.audio.numpy_transforms.save_wav") as mock_save:
                returned = tts.tts_to_file("Hi", file_path=out_path)
            mock_save.assert_called_once_with(
                wav=fake_wav, path=out_path, sample_rate=MiniMaxTTS.SAMPLE_RATE
            )
            self.assertEqual(returned, out_path)
        finally:
            if os.path.exists(out_path):
                os.unlink(out_path)


class TestMiniMaxTTSConstants(unittest.TestCase):
    def test_models_list(self):
        self.assertIn("speech-2.8-hd", MINIMAX_MODELS)
        self.assertIn("speech-2.8-turbo", MINIMAX_MODELS)

    def test_voices_list(self):
        self.assertGreater(len(MINIMAX_VOICES), 0)
        for v in MINIMAX_VOICES:
            self.assertIsInstance(v, str)

    def test_sample_rate(self):
        self.assertEqual(MiniMaxTTS.SAMPLE_RATE, 32000)


# ---------------------------------------------------------------------------
# Integration tests (skipped when MINIMAX_API_KEY is absent)
# ---------------------------------------------------------------------------

MINIMAX_API_KEY = os.environ.get("MINIMAX_API_KEY", "")


@unittest.skipUnless(MINIMAX_API_KEY, "MINIMAX_API_KEY not set — skipping integration tests")
class TestMiniMaxTTSIntegration(unittest.TestCase):
    """Live API integration tests.

    These tests are skipped automatically in CI unless MINIMAX_API_KEY is
    exported in the environment.
    """

    def setUp(self):
        self.tts = MiniMaxTTS(api_key=MINIMAX_API_KEY)

    def test_tts_returns_non_empty_array(self):
        wav = self.tts.tts("Hello, MiniMax!")
        self.assertIsInstance(wav, np.ndarray)
        self.assertGreater(len(wav), 100)
        self.assertEqual(wav.dtype, np.float32)

    def test_tts_to_file_creates_wav(self):
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            out_path = f.name
        try:
            result = self.tts.tts_to_file("Integration test.", file_path=out_path)
            self.assertEqual(result, out_path)
            self.assertTrue(os.path.exists(out_path))
            self.assertGreater(os.path.getsize(out_path), 0)
        finally:
            if os.path.exists(out_path):
                os.unlink(out_path)

    def test_tts_with_different_voice(self):
        wav = self.tts.tts("Testing voice options.", voice="English_Lucky_Robot")
        self.assertIsInstance(wav, np.ndarray)
        self.assertGreater(len(wav), 100)

    def test_tts_turbo_model(self):
        tts_turbo = MiniMaxTTS(model="speech-2.8-turbo", api_key=MINIMAX_API_KEY)
        wav = tts_turbo.tts("Testing turbo model.")
        self.assertIsInstance(wav, np.ndarray)
        self.assertGreater(len(wav), 100)


if __name__ == "__main__":
    unittest.main()
