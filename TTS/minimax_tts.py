"""
MiniMax TTS integration for coqui-ai/TTS.

Provides a lightweight TTS interface backed by the MiniMax T2A v2 API,
compatible with coqui-ai/TTS audio utilities.

Usage::

    from TTS.minimax_tts import MiniMaxTTS

    tts = MiniMaxTTS(api_key="your-api-key")
    wav = tts.tts("Hello world!")
    tts.tts_to_file("Hello world!", file_path="output.wav")

Environment:
    MINIMAX_API_KEY: MiniMax API key (used when *api_key* argument is omitted).
"""

import io
import json
import os
import urllib.request
import urllib.error
from typing import List, Optional

import numpy as np


MINIMAX_VOICES: List[str] = [
    "English_Graceful_Lady",
    "English_Insightful_Speaker",
    "English_radiant_girl",
    "English_Persuasive_Man",
    "English_Lucky_Robot",
    "English_expressive_narrator",
]

MINIMAX_MODELS: List[str] = [
    "speech-2.8-hd",
    "speech-2.8-turbo",
]

MINIMAX_SAMPLE_RATE: int = 32000
MINIMAX_DEFAULT_BASE_URL: str = "https://api.minimax.io"


class MiniMaxTTS:
    """MiniMax TTS API client compatible with coqui-ai/TTS.

    Calls the MiniMax T2A v2 endpoint and returns audio as a float32
    numpy array at :attr:`SAMPLE_RATE` Hz, ready to be passed to
    :func:`TTS.utils.audio.numpy_transforms.save_wav` or any other
    coqui audio utility.

    Args:
        model (str): MiniMax TTS model to use.
            One of ``"speech-2.8-hd"`` (recommended) or
            ``"speech-2.8-turbo"`` (faster). Defaults to ``"speech-2.8-hd"``.
        api_key (str, optional): MiniMax API key. Falls back to the
            ``MINIMAX_API_KEY`` environment variable when omitted.
        base_url (str): MiniMax API base URL.
            Defaults to ``"https://api.minimax.io"``.

    Raises:
        ValueError: If *model* is not recognised or *api_key* cannot be
            resolved.

    Example::

        >>> tts = MiniMaxTTS()
        >>> wav = tts.tts("Hello, this is a test.")
        >>> tts.tts_to_file("Hello, world!", file_path="output.wav")
    """

    SAMPLE_RATE: int = MINIMAX_SAMPLE_RATE

    def __init__(
        self,
        model: str = "speech-2.8-hd",
        api_key: Optional[str] = None,
        base_url: str = MINIMAX_DEFAULT_BASE_URL,
    ) -> None:
        if model not in MINIMAX_MODELS:
            raise ValueError(
                f"Unknown model '{model}'. Available models: {MINIMAX_MODELS}"
            )
        resolved_key = api_key or os.environ.get("MINIMAX_API_KEY", "")
        if not resolved_key:
            raise ValueError(
                "MiniMax API key is required. "
                "Pass it via the 'api_key' argument or set the "
                "MINIMAX_API_KEY environment variable."
            )
        self.model = model
        self._api_key = resolved_key
        self.base_url = base_url.rstrip("/")

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    @property
    def voices(self) -> List[str]:
        """Return the list of available MiniMax voice IDs."""
        return list(MINIMAX_VOICES)

    # ------------------------------------------------------------------
    # Core synthesis methods
    # ------------------------------------------------------------------

    def tts(
        self,
        text: str,
        voice: str = "English_Graceful_Lady",
        speed: float = 1.0,
        volume: float = 1.0,
        pitch: int = 0,
    ) -> np.ndarray:
        """Convert text to speech using the MiniMax T2A v2 API.

        Args:
            text (str): Input text to synthesise.
            voice (str): Voice ID. Defaults to ``"English_Graceful_Lady"``.
                See :data:`MINIMAX_VOICES` for all supported IDs.
            speed (float): Speech rate multiplier in ``[0.5, 2.0]``.
                Defaults to ``1.0``.
            volume (float): Loudness multiplier in ``[0.1, 10.0]``.
                Defaults to ``1.0``.
            pitch (int): Pitch shift in semitones ``[-12, 12]``.
                Defaults to ``0``.

        Returns:
            numpy.ndarray: Audio waveform as a float32 array sampled at
            :attr:`SAMPLE_RATE` Hz.

        Raises:
            urllib.error.HTTPError: On API HTTP errors.
            RuntimeError: If the API returns no audio data.
        """
        mp3_bytes = self._call_api(
            text=text, voice=voice, speed=speed, volume=volume, pitch=pitch
        )
        return self._mp3_to_numpy(mp3_bytes)

    def tts_to_file(
        self,
        text: str,
        voice: str = "English_Graceful_Lady",
        speed: float = 1.0,
        volume: float = 1.0,
        pitch: int = 0,
        file_path: str = "output.wav",
    ) -> str:
        """Convert text to speech and save to a WAV file.

        Args:
            text (str): Input text to synthesise.
            voice (str): Voice ID. Defaults to ``"English_Graceful_Lady"``.
            speed (float): Speech rate multiplier. Defaults to ``1.0``.
            volume (float): Loudness multiplier. Defaults to ``1.0``.
            pitch (int): Pitch shift in semitones. Defaults to ``0``.
            file_path (str): Destination WAV path. Defaults to
                ``"output.wav"``.

        Returns:
            str: Absolute path of the saved WAV file.
        """
        from TTS.utils.audio.numpy_transforms import save_wav  # lazy import

        wav = self.tts(
            text=text, voice=voice, speed=speed, volume=volume, pitch=pitch
        )
        save_wav(wav=wav, path=file_path, sample_rate=self.SAMPLE_RATE)
        return file_path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _call_api(
        self,
        text: str,
        voice: str,
        speed: float,
        volume: float,
        pitch: int,
    ) -> bytes:
        """Call the MiniMax T2A v2 endpoint and collect the MP3 audio.

        The endpoint streams SSE events, each carrying a ``data.audio``
        field that contains a hex-encoded MP3 fragment. This method
        collects all fragments and returns the concatenated bytes.

        Returns:
            bytes: Raw MP3 audio.

        Raises:
            RuntimeError: If no audio data is received.
        """
        url = f"{self.base_url}/v1/t2a_v2"
        payload = json.dumps(
            {
                "model": self.model,
                "text": text,
                "stream": True,
                "voice_setting": {
                    "voice_id": voice,
                    "speed": speed,
                    "vol": volume,
                    "pitch": pitch,
                },
                "audio_setting": {
                    "sample_rate": self.SAMPLE_RATE,
                    "bitrate": 128000,
                    "format": "mp3",
                    "channel": 1,
                },
            }
        ).encode()

        req = urllib.request.Request(
            url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
        )

        audio_chunks: List[bytes] = []
        buf = b""
        with urllib.request.urlopen(req) as resp:
            while True:
                block = resp.read(8192)
                if not block:
                    break
                buf += block
                *lines, buf = buf.split(b"\n")
                for line in lines:
                    line = line.strip()
                    if not line.startswith(b"data:"):
                        continue
                    json_str = line[5:].strip()
                    if not json_str or json_str == b"[DONE]":
                        continue
                    try:
                        event = json.loads(json_str)
                        hex_audio = event.get("data", {}).get("audio", "")
                        if hex_audio:
                            audio_chunks.append(bytes.fromhex(hex_audio))
                    except (json.JSONDecodeError, ValueError):
                        pass

        if not audio_chunks:
            raise RuntimeError(
                "MiniMax TTS API returned no audio data. "
                "Check your API key and request parameters."
            )
        return b"".join(audio_chunks)

    @staticmethod
    def _mp3_to_numpy(mp3_bytes: bytes) -> np.ndarray:
        """Decode MP3 bytes into a float32 numpy waveform.

        Attempts to use *torchaudio* (bundled with coqui-ai/TTS) first,
        then falls back to *soundfile* (also in requirements) via ffmpeg.

        Args:
            mp3_bytes (bytes): Raw MP3 audio data.

        Returns:
            numpy.ndarray: Mono float32 waveform.

        Raises:
            ImportError: If neither torchaudio nor soundfile is available.
        """
        # Preferred: torchaudio (already a coqui-ai/TTS dependency)
        try:
            import torch
            import torchaudio

            buf = io.BytesIO(mp3_bytes)
            wav_tensor, _ = torchaudio.load(buf, format="mp3")
            if wav_tensor.shape[0] > 1:
                wav_tensor = wav_tensor.mean(dim=0, keepdim=True)
            return wav_tensor.squeeze(0).numpy().astype(np.float32)
        except Exception:  # noqa: BLE001
            pass

        # Fallback: soundfile (also a coqui-ai/TTS dependency, needs ffmpeg for mp3)
        try:
            import soundfile as sf

            buf = io.BytesIO(mp3_bytes)
            wav, _ = sf.read(buf, dtype="float32")
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            return wav.astype(np.float32)
        except Exception:  # noqa: BLE001
            pass

        raise ImportError(
            "Could not decode MiniMax MP3 audio. "
            "Ensure that torchaudio or soundfile (with ffmpeg) is installed.\n"
            "  pip install torchaudio\n"
            "  pip install soundfile"
        )
