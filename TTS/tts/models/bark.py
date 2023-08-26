import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
from coqpit import Coqpit
from encodec import EncodecModel
from transformers import BertTokenizer

from TTS.tts.layers.bark.inference_funcs import (
    codec_decode,
    generate_coarse,
    generate_fine,
    generate_text_semantic,
    generate_voice,
    load_voice,
)
from TTS.tts.layers.bark.load_model import load_model
from TTS.tts.layers.bark.model import GPT
from TTS.tts.layers.bark.model_fine import FineGPT
from TTS.tts.models.base_tts import BaseTTS


@dataclass
class BarkAudioConfig(Coqpit):
    sample_rate: int = 24000
    output_sample_rate: int = 24000


class Bark(BaseTTS):
    def __init__(
        self,
        config: Coqpit,
        tokenizer: BertTokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased"),
    ) -> None:
        super().__init__(config=config, ap=None, tokenizer=None, speaker_manager=None, language_manager=None)
        self.config.num_chars = len(tokenizer)
        self.tokenizer = tokenizer
        self.semantic_model = GPT(config.semantic_config)
        self.coarse_model = GPT(config.coarse_config)
        self.fine_model = FineGPT(config.fine_config)
        self.encodec = EncodecModel.encodec_model_24khz()
        self.encodec.set_target_bandwidth(6.0)

    @property
    def device(self):
        return next(self.parameters()).device

    def load_bark_models(self):
        self.semantic_model, self.config = load_model(
            ckpt_path=self.config.LOCAL_MODEL_PATHS["text"], device=self.device, config=self.config, model_type="text"
        )
        self.coarse_model, self.config = load_model(
            ckpt_path=self.config.LOCAL_MODEL_PATHS["coarse"],
            device=self.device,
            config=self.config,
            model_type="coarse",
        )
        self.fine_model, self.config = load_model(
            ckpt_path=self.config.LOCAL_MODEL_PATHS["fine"], device=self.device, config=self.config, model_type="fine"
        )

    def train_step(
        self,
    ):
        pass

    def text_to_semantic(
        self,
        text: str,
        history_prompt: Optional[str] = None,
        temp: float = 0.7,
        base=None,
        allow_early_stop=True,
        **kwargs,
    ):
        """Generate semantic array from text.

        Args:
            text: text to be turned into audio
            history_prompt: history choice for audio cloning
            temp: generation temperature (1.0 more diverse, 0.0 more conservative)

        Returns:
            numpy semantic array to be fed into `semantic_to_waveform`
        """
        x_semantic = generate_text_semantic(
            text,
            self,
            history_prompt=history_prompt,
            temp=temp,
            base=base,
            allow_early_stop=allow_early_stop,
            **kwargs,
        )
        return x_semantic

    def semantic_to_waveform(
        self,
        semantic_tokens: np.ndarray,
        history_prompt: Optional[str] = None,
        temp: float = 0.7,
        base=None,
    ):
        """Generate audio array from semantic input.

        Args:
            semantic_tokens: semantic token output from `text_to_semantic`
            history_prompt: history choice for audio cloning
            temp: generation temperature (1.0 more diverse, 0.0 more conservative)

        Returns:
            numpy audio array at sample frequency 24khz
        """
        x_coarse_gen = generate_coarse(
            semantic_tokens,
            self,
            history_prompt=history_prompt,
            temp=temp,
            base=base,
        )
        x_fine_gen = generate_fine(
            x_coarse_gen,
            self,
            history_prompt=history_prompt,
            temp=0.5,
            base=base,
        )
        audio_arr = codec_decode(x_fine_gen, self)
        return audio_arr, x_coarse_gen, x_fine_gen

    def generate_audio(
        self,
        text: str,
        history_prompt: Optional[str] = None,
        text_temp: float = 0.7,
        waveform_temp: float = 0.7,
        base=None,
        allow_early_stop=True,
        **kwargs,
    ):
        """Generate audio array from input text.

        Args:
            text: text to be turned into audio
            history_prompt: history choice for audio cloning
            text_temp: generation temperature (1.0 more diverse, 0.0 more conservative)
            waveform_temp: generation temperature (1.0 more diverse, 0.0 more conservative)

        Returns:
            numpy audio array at sample frequency 24khz
        """
        x_semantic = self.text_to_semantic(
            text,
            history_prompt=history_prompt,
            temp=text_temp,
            base=base,
            allow_early_stop=allow_early_stop,
            **kwargs,
        )
        audio_arr, c, f = self.semantic_to_waveform(
            x_semantic, history_prompt=history_prompt, temp=waveform_temp, base=base
        )
        return audio_arr, [x_semantic, c, f]

    def generate_voice(self, audio, speaker_id, voice_dir):
        """Generate a voice from the given audio and text.

        Args:
            audio (str): Path to the audio file.
            speaker_id (str): Speaker name.
            voice_dir (str): Path to the directory to save the generate voice.
        """
        if voice_dir is not None:
            voice_dirs = [voice_dir]
            try:
                _ = load_voice(speaker_id, voice_dirs)
            except (KeyError, FileNotFoundError):
                output_path = os.path.join(voice_dir, speaker_id + ".npz")
                os.makedirs(voice_dir, exist_ok=True)
                generate_voice(audio, self, output_path)

    def _set_voice_dirs(self, voice_dirs):
        def_voice_dir = None
        if isinstance(self.config.DEF_SPEAKER_DIR, str):
            os.makedirs(self.config.DEF_SPEAKER_DIR, exist_ok=True)
            if os.path.isdir(self.config.DEF_SPEAKER_DIR):
                def_voice_dir = self.config.DEF_SPEAKER_DIR
        _voice_dirs = [def_voice_dir] if def_voice_dir is not None else []
        if voice_dirs is not None:
            if isinstance(voice_dirs, str):
                voice_dirs = [voice_dirs]
            _voice_dirs = voice_dirs + _voice_dirs
        return _voice_dirs

    # TODO: remove config from synthesize
    def synthesize(
        self, text, config, speaker_id="random", voice_dirs=None, **kwargs
    ):  # pylint: disable=unused-argument
        """Synthesize speech with the given input text.

        Args:
            text (str): Input text.
            config (BarkConfig): Config with inference parameters.
            speaker_id (str): One of the available speaker names. If `random`, it generates a random speaker.
            speaker_wav (str): Path to the speaker audio file for cloning a new voice. It is cloned and saved in
                `voice_dirs` with the name `speaker_id`. Defaults to None.
            voice_dirs (List[str]): List of paths that host reference audio files for speakers. Defaults to None.
            **kwargs: Model specific inference settings used by `generate_audio()` and `TTS.tts.layers.bark.inference_funcs.generate_text_semantic().

        Returns:
            A dictionary of the output values with `wav` as output waveform, `deterministic_seed` as seed used at inference,
            `text_input` as text token IDs after tokenizer, `voice_samples` as samples used for cloning, `conditioning_latents`
            as latents used at inference.

        """
        speaker_id = "random" if speaker_id is None else speaker_id
        voice_dirs = self._set_voice_dirs(voice_dirs)
        history_prompt = load_voice(self, speaker_id, voice_dirs)
        outputs = self.generate_audio(text, history_prompt=history_prompt, **kwargs)
        return_dict = {
            "wav": outputs[0],
            "text_inputs": text,
        }

        return return_dict

    def eval_step(self):
        ...

    def forward(self):
        ...

    def inference(self):
        ...

    @staticmethod
    def init_from_config(config: "BarkConfig", **kwargs):  # pylint: disable=unused-argument
        return Bark(config)

    # pylint: disable=unused-argument, redefined-builtin
    def load_checkpoint(
        self,
        config,
        checkpoint_dir,
        text_model_path=None,
        coarse_model_path=None,
        fine_model_path=None,
        hubert_model_path=None,
        hubert_tokenizer_path=None,
        eval=False,
        strict=True,
        **kwargs,
    ):
        """Load a model checkpoints from a directory. This model is with multiple checkpoint files and it
        expects to have all the files to be under the given `checkpoint_dir` with the rigth names.
        If eval is True, set the model to eval mode.

        Args:
            config (TortoiseConfig): The model config.
            checkpoint_dir (str): The directory where the checkpoints are stored.
            ar_checkpoint_path (str, optional): The path to the autoregressive checkpoint. Defaults to None.
            diff_checkpoint_path (str, optional): The path to the diffusion checkpoint. Defaults to None.
            clvp_checkpoint_path (str, optional): The path to the CLVP checkpoint. Defaults to None.
            vocoder_checkpoint_path (str, optional): The path to the vocoder checkpoint. Defaults to None.
            eval (bool, optional): Whether to set the model to eval mode. Defaults to False.
            strict (bool, optional): Whether to load the model strictly. Defaults to True.
        """
        text_model_path = text_model_path or os.path.join(checkpoint_dir, "text_2.pt")
        coarse_model_path = coarse_model_path or os.path.join(checkpoint_dir, "coarse_2.pt")
        fine_model_path = fine_model_path or os.path.join(checkpoint_dir, "fine_2.pt")
        hubert_model_path = hubert_model_path or os.path.join(checkpoint_dir, "hubert.pt")
        hubert_tokenizer_path = hubert_tokenizer_path or os.path.join(checkpoint_dir, "tokenizer.pth")

        self.config.LOCAL_MODEL_PATHS["text"] = text_model_path
        self.config.LOCAL_MODEL_PATHS["coarse"] = coarse_model_path
        self.config.LOCAL_MODEL_PATHS["fine"] = fine_model_path
        self.config.LOCAL_MODEL_PATHS["hubert"] = hubert_model_path
        self.config.LOCAL_MODEL_PATHS["hubert_tokenizer"] = hubert_tokenizer_path

        self.load_bark_models()

        if eval:
            self.eval()
