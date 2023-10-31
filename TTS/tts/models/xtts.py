import os
from contextlib import contextmanager
from dataclasses import dataclass

import librosa
import torch
import torch.nn.functional as F
import torchaudio
from coqpit import Coqpit

from TTS.tts.layers.tortoise.audio_utils import denormalize_tacotron_mel, wav_to_univnet_mel
from TTS.tts.layers.tortoise.diffusion_decoder import DiffusionTts
from TTS.tts.layers.xtts.diffusion import SpacedDiffusion, get_named_beta_schedule, space_timesteps
from TTS.tts.layers.xtts.gpt import GPT
from TTS.tts.layers.xtts.hifigan_decoder import HifiDecoder
from TTS.tts.layers.xtts.stream_generator import init_stream_support
from TTS.tts.layers.xtts.tokenizer import VoiceBpeTokenizer
from TTS.tts.layers.xtts.vocoder import UnivNetGenerator
from TTS.tts.models.base_tts import BaseTTS
from TTS.utils.io import load_fsspec

init_stream_support()


def wav_to_mel_cloning(
    wav, mel_norms_file="../experiments/clips_mel_norms.pth", mel_norms=None, device=torch.device("cpu")
):
    """
    Convert waveform to mel-spectrogram with hard-coded parameters for cloning.

    Args:
        wav (torch.Tensor): Input waveform tensor.
        mel_norms_file (str): Path to mel-spectrogram normalization file.
        mel_norms (torch.Tensor): Mel-spectrogram normalization tensor.
        device (torch.device): Device to use for computation.

    Returns:
        torch.Tensor: Mel-spectrogram tensor.
    """
    mel_stft = torchaudio.transforms.MelSpectrogram(
        n_fft=4096,
        hop_length=1024,
        win_length=4096,
        power=2,
        normalized=False,
        sample_rate=22050,
        f_min=0,
        f_max=8000,
        n_mels=80,
        norm="slaney",
    ).to(device)
    wav = wav.to(device)
    mel = mel_stft(wav)
    mel = torch.log(torch.clamp(mel, min=1e-5))
    if mel_norms is None:
        mel_norms = torch.load(mel_norms_file, map_location=device)
    mel = mel / mel_norms.unsqueeze(0).unsqueeze(-1)
    return mel


def pad_or_truncate(t, length):
    """
    Ensure a given tensor t has a specified sequence length by either padding it with zeros or clipping it.

    Args:
        t (torch.Tensor): The input tensor to be padded or truncated.
        length (int): The desired length of the tensor.

    Returns:
        torch.Tensor: The padded or truncated tensor.
    """
    tp = t[..., :length]
    if t.shape[-1] == length:
        tp = t
    elif t.shape[-1] < length:
        tp = F.pad(t, (0, length - t.shape[-1]))
    return tp


def load_discrete_vocoder_diffuser(
    trained_diffusion_steps=4000,
    desired_diffusion_steps=200,
    cond_free=True,
    cond_free_k=1,
    sampler="ddim",
):
    """
    Load a GaussianDiffusion instance configured for use as a decoder.

    Args:
        trained_diffusion_steps (int): The number of diffusion steps used during training.
        desired_diffusion_steps (int): The number of diffusion steps to use during inference.
        cond_free (bool): Whether to use a conditioning-free model.
        cond_free_k (int): The number of samples to use for conditioning-free models.
        sampler (str): The name of the sampler to use.

    Returns:
        A SpacedDiffusion instance configured with the given parameters.
    """
    return SpacedDiffusion(
        use_timesteps=space_timesteps(trained_diffusion_steps, [desired_diffusion_steps]),
        model_mean_type="epsilon",
        model_var_type="learned_range",
        loss_type="mse",
        betas=get_named_beta_schedule("linear", trained_diffusion_steps),
        conditioning_free=cond_free,
        conditioning_free_k=cond_free_k,
        sampler=sampler,
    )


def do_spectrogram_diffusion(
    diffusion_model,
    diffuser,
    latents,
    conditioning_latents,
    temperature=1,
):
    """
    Generate a mel-spectrogram using a diffusion model and a diffuser.

    Args:
        diffusion_model (nn.Module): A diffusion model that converts from 22kHz spectrogram codes to a 24kHz spectrogram signal.
        diffuser (Diffuser): A diffuser that generates a mel-spectrogram from noise.
        latents (torch.Tensor): A tensor of shape (batch_size, seq_len, code_size) containing the input spectrogram codes.
        conditioning_latents (torch.Tensor): A tensor of shape (batch_size, code_size) containing the conditioning codes.
        temperature (float, optional): The temperature of the noise used by the diffuser. Defaults to 1.

    Returns:
        torch.Tensor: A tensor of shape (batch_size, mel_channels, mel_seq_len) containing the generated mel-spectrogram.
    """
    with torch.no_grad():
        output_seq_len = (
            latents.shape[1] * 4 * 24000 // 22050
        )  # This diffusion model converts from 22kHz spectrogram codes to a 24kHz spectrogram signal.
        output_shape = (latents.shape[0], 100, output_seq_len)
        precomputed_embeddings = diffusion_model.timestep_independent(
            latents, conditioning_latents, output_seq_len, False
        )

        noise = torch.randn(output_shape, device=latents.device) * temperature
        mel = diffuser.sample_loop(
            diffusion_model,
            output_shape,
            noise=noise,
            model_kwargs={"precomputed_aligned_embeddings": precomputed_embeddings},
            progress=False,
        )
        return denormalize_tacotron_mel(mel)[:, :, :output_seq_len]


@dataclass
class XttsAudioConfig(Coqpit):
    """
    Configuration class for audio-related parameters in the XTTS model.

    Args:
        sample_rate (int): The sample rate in which the GPT operates.
        diffusion_sample_rate (int): The sample rate of the diffusion audio waveform.
        output_sample_rate (int): The sample rate of the output audio waveform.
    """

    sample_rate: int = 22050
    diffusion_sample_rate: int = 24000
    output_sample_rate: int = 24000


@dataclass
class XttsArgs(Coqpit):
    """A dataclass to represent XTTS model arguments that define the model structure.

    Args:
        gpt_batch_size (int): The size of the auto-regressive batch.
        enable_redaction (bool, optional): Whether to enable redaction. Defaults to True.
        kv_cache (bool, optional): Whether to use the kv_cache. Defaults to True.
        gpt_checkpoint (str, optional): The checkpoint for the autoregressive model. Defaults to None.
        clvp_checkpoint (str, optional): The checkpoint for the ConditionalLatentVariablePerseq model. Defaults to None.
        decoder_checkpoint (str, optional): The checkpoint for the DiffTTS model. Defaults to None.
        num_chars (int, optional): The maximum number of characters to generate. Defaults to 255.
        use_hifigan (bool, optional): Whether to use hifigan or diffusion + univnet as a decoder. Defaults to True.

        For GPT model:
        ar_max_audio_tokens (int, optional): The maximum mel tokens for the autoregressive model. Defaults to 604.
        ar_max_text_tokens (int, optional): The maximum text tokens for the autoregressive model. Defaults to 402.
        ar_max_prompt_tokens (int, optional): The maximum prompt tokens or the autoregressive model. Defaults to 70.
        ar_layers (int, optional): The number of layers for the autoregressive model. Defaults to 30.
        ar_n_model_channels (int, optional): The model dimension for the autoregressive model. Defaults to 1024.
        ar_n_heads (int, optional): The number of heads for the autoregressive model. Defaults to 16.
        ar_number_text_tokens (int, optional): The number of text tokens for the autoregressive model. Defaults to 255.
        ar_start_text_token (int, optional): The start text token for the autoregressive model. Defaults to 255.
        gpt_checkpointing (bool, optional): Whether to use checkpointing for the autoregressive model. Defaults to False.
        ar_train_solo_embeddings (bool, optional): Whether to train embeddings for the autoregressive model. Defaults to False.

        For DiffTTS model:
        diff_model_channels (int, optional): The number of channels for the DiffTTS model. Defaults to 1024.
        diff_num_layers (int, optional): The number of layers for the DiffTTS model. Defaults to 10.
        diff_in_channels (int, optional): The input channels for the DiffTTS model. Defaults to 100.
        diff_out_channels (int, optional): The output channels for the DiffTTS model. Defaults to 200.
        diff_in_latent_channels (int, optional): The input latent channels for the DiffTTS model. Defaults to 1024.
        diff_in_tokens (int, optional): The input tokens for the DiffTTS model. Defaults to 8193.
        diff_dropout (int, optional): The dropout percentage for the DiffTTS model. Defaults to 0.
        diff_use_fp16 (bool, optional): Whether to use fp16 for the DiffTTS model. Defaults to False.
        diff_num_heads (int, optional): The number of heads for the DiffTTS model. Defaults to 16.
        diff_layer_drop (int, optional): The layer dropout percentage for the DiffTTS model. Defaults to 0.
        diff_unconditioned_percentage (int, optional): The percentage of unconditioned inputs for the DiffTTS model. Defaults to 0.
    """

    gpt_batch_size: int = 1
    enable_redaction: bool = False
    kv_cache: bool = True
    gpt_checkpoint: str = None
    clvp_checkpoint: str = None
    decoder_checkpoint: str = None
    num_chars: int = 255
    use_hifigan: bool = True
    use_ne_hifigan: bool = False

    # XTTS GPT Encoder params
    tokenizer_file: str = ""
    gpt_max_audio_tokens: int = 605
    gpt_max_text_tokens: int = 402
    gpt_max_prompt_tokens: int = 70
    gpt_layers: int = 30
    gpt_n_model_channels: int = 1024
    gpt_n_heads: int = 16
    gpt_number_text_tokens: int = None
    gpt_start_text_token: int = None
    gpt_stop_text_token: int = None
    gpt_num_audio_tokens: int = 8194
    gpt_start_audio_token: int = 8192
    gpt_stop_audio_token: int = 8193

    # Diffusion Decoder params
    diff_model_channels: int = 1024
    diff_num_layers: int = 10
    diff_in_channels: int = 100
    diff_out_channels: int = 200
    diff_in_latent_channels: int = 1024
    diff_in_tokens: int = 8193
    diff_dropout: int = 0
    diff_use_fp16: bool = False
    diff_num_heads: int = 16
    diff_layer_drop: int = 0
    diff_unconditioned_percentage: int = 0

    # HifiGAN Decoder params
    input_sample_rate: int = 22050
    output_sample_rate: int = 24000
    output_hop_length: int = 256
    ar_mel_length_compression: int = 1024
    decoder_input_dim: int = 1024
    d_vector_dim: int = 512
    cond_d_vector_in_each_upsampling_layer: bool = True

    # constants
    duration_const: int = 102400


class Xtts(BaseTTS):
    """ⓍTTS model implementation.

    ❗ Currently it only supports inference.

    Examples:
        >>> from TTS.tts.configs.xtts_config import XttsConfig
        >>> from TTS.tts.models.xtts import Xtts
        >>> config = XttsConfig()
        >>> model = Xtts.inif_from_config(config)
        >>> model.load_checkpoint(config, checkpoint_dir="paths/to/models_dir/", eval=True)
    """

    def __init__(self, config: Coqpit):
        super().__init__(config, ap=None, tokenizer=None)
        self.mel_stats_path = None
        self.config = config
        self.gpt_checkpoint = self.args.gpt_checkpoint
        self.decoder_checkpoint = self.args.decoder_checkpoint  # TODO: check if this is even needed
        self.models_dir = config.model_dir
        self.gpt_batch_size = self.args.gpt_batch_size

        self.tokenizer = VoiceBpeTokenizer()
        self.gpt = None
        self.init_models()
        self.register_buffer("mel_stats", torch.ones(80))

    def init_models(self):
        """Initialize the models. We do it here since we need to load the tokenizer first."""
        if self.tokenizer.tokenizer is not None:
            self.args.gpt_number_text_tokens = self.tokenizer.get_number_tokens()
            self.args.gpt_start_text_token = self.tokenizer.tokenizer.token_to_id("[START]")
            self.args.gpt_stop_text_token = self.tokenizer.tokenizer.token_to_id("[STOP]")

        if self.args.gpt_number_text_tokens:
            self.gpt = GPT(
                layers=self.args.gpt_layers,
                model_dim=self.args.gpt_n_model_channels,
                start_text_token=self.args.gpt_start_text_token,
                stop_text_token=self.args.gpt_stop_text_token,
                heads=self.args.gpt_n_heads,
                max_text_tokens=self.args.gpt_max_text_tokens,
                max_mel_tokens=self.args.gpt_max_audio_tokens,
                max_prompt_tokens=self.args.gpt_max_prompt_tokens,
                number_text_tokens=self.args.gpt_number_text_tokens,
                num_audio_tokens=self.args.gpt_num_audio_tokens,
                start_audio_token=self.args.gpt_start_audio_token,
                stop_audio_token=self.args.gpt_stop_audio_token,
            )

        if self.args.use_hifigan:
            self.hifigan_decoder = HifiDecoder(
                input_sample_rate=self.args.input_sample_rate,
                output_sample_rate=self.args.output_sample_rate,
                output_hop_length=self.args.output_hop_length,
                ar_mel_length_compression=self.args.ar_mel_length_compression,
                decoder_input_dim=self.args.decoder_input_dim,
                d_vector_dim=self.args.d_vector_dim,
                cond_d_vector_in_each_upsampling_layer=self.args.cond_d_vector_in_each_upsampling_layer,
            )

        if self.args.use_ne_hifigan:
            self.ne_hifigan_decoder = HifiDecoder(
                input_sample_rate=self.args.input_sample_rate,
                output_sample_rate=self.args.output_sample_rate,
                output_hop_length=self.args.output_hop_length,
                ar_mel_length_compression=self.args.ar_mel_length_compression,
                decoder_input_dim=self.args.decoder_input_dim,
                d_vector_dim=self.args.d_vector_dim,
                cond_d_vector_in_each_upsampling_layer=self.args.cond_d_vector_in_each_upsampling_layer,
            )

        if not (self.args.use_hifigan or self.args.use_ne_hifigan):
            self.diffusion_decoder = DiffusionTts(
                model_channels=self.args.diff_model_channels,
                num_layers=self.args.diff_num_layers,
                in_channels=self.args.diff_in_channels,
                out_channels=self.args.diff_out_channels,
                in_latent_channels=self.args.diff_in_latent_channels,
                in_tokens=self.args.diff_in_tokens,
                dropout=self.args.diff_dropout,
                use_fp16=self.args.diff_use_fp16,
                num_heads=self.args.diff_num_heads,
                layer_drop=self.args.diff_layer_drop,
                unconditioned_percentage=self.args.diff_unconditioned_percentage,
            )
            self.vocoder = UnivNetGenerator()

    @property
    def device(self):
        return next(self.parameters()).device

    @torch.inference_mode()
    def get_gpt_cond_latents(self, audio, sr, length: int = 3):
        """Compute the conditioning latents for the GPT model from the given audio.

        Args:
            audio_path (str): Path to the audio file.
            length (int): Length of the audio in seconds. Defaults to 3.
        """

        audio_22k = torchaudio.functional.resample(audio, sr, 22050)
        audio_22k = audio_22k[:, : 22050 * length]
        mel = wav_to_mel_cloning(audio_22k, mel_norms=self.mel_stats.cpu())
        cond_latent = self.gpt.get_style_emb(mel.to(self.device))
        return cond_latent.transpose(1, 2)

    @torch.inference_mode()
    def get_diffusion_cond_latents(self, audio, sr):
        from math import ceil

        diffusion_conds = []
        CHUNK_SIZE = 102400
        audio_24k = torchaudio.functional.resample(audio, sr, 24000)
        for chunk in range(ceil(audio_24k.shape[1] / CHUNK_SIZE)):
            current_sample = audio_24k[:, chunk * CHUNK_SIZE : (chunk + 1) * CHUNK_SIZE]
            current_sample = pad_or_truncate(current_sample, CHUNK_SIZE)
            cond_mel = wav_to_univnet_mel(
                current_sample.to(self.device),
                do_normalization=False,
                device=self.device,
            )
            diffusion_conds.append(cond_mel)
        diffusion_conds = torch.stack(diffusion_conds, dim=1)
        diffusion_latent = self.diffusion_decoder.get_conditioning(diffusion_conds)
        return diffusion_latent

    @torch.inference_mode()
    def get_speaker_embedding(self, audio, sr):
        audio_16k = torchaudio.functional.resample(audio, sr, 16000)
        return (
            self.hifigan_decoder.speaker_encoder.forward(audio_16k.to(self.device), l2_norm=True)
            .unsqueeze(-1)
            .to(self.device)
        )

    @torch.inference_mode()
    def get_conditioning_latents(
        self,
        audio_path,
        gpt_cond_len=6,
        max_ref_length=10,
        librosa_trim_db=None,
        sound_norm_refs=False,
    ):
        speaker_embedding = None
        diffusion_cond_latents = None

        audio, sr = torchaudio.load(audio_path)
        audio = audio[:, : sr * max_ref_length].to(self.device)
        if audio.shape[0] > 1:
            audio = audio.mean(0, keepdim=True)
        if sound_norm_refs:
            audio = (audio / torch.abs(audio).max()) * 0.75
        if librosa_trim_db is not None:
            audio = librosa.effects.trim(audio, top_db=librosa_trim_db)[0]

        if self.args.use_hifigan or self.args.use_ne_hifigan:
            speaker_embedding = self.get_speaker_embedding(audio, sr)
        else:
            diffusion_cond_latents = self.get_diffusion_cond_latents(audio, sr)
        gpt_cond_latents = self.get_gpt_cond_latents(audio, sr, length=gpt_cond_len)  # [1, 1024, T]
        return gpt_cond_latents, diffusion_cond_latents, speaker_embedding

    def synthesize(self, text, config, speaker_wav, language, **kwargs):
        """Synthesize speech with the given input text.

        Args:
            text (str): Input text.
            config (XttsConfig): Config with inference parameters.
            speaker_wav (str): Path to the speaker audio file for cloning.
            language (str): Language ID of the speaker.
            **kwargs: Inference settings. See `inference()`.

        Returns:
            A dictionary of the output values with `wav` as output waveform, `deterministic_seed` as seed used at inference,
            `text_input` as text token IDs after tokenizer, `voice_samples` as samples used for cloning, `conditioning_latents`
            as latents used at inference.

        """

        # Make the synthesizer happy 🥳
        if isinstance(speaker_wav, list):
            speaker_wav = speaker_wav[0]

        return self.inference_with_config(text, config, ref_audio_path=speaker_wav, language=language, **kwargs)

    def inference_with_config(self, text, config, ref_audio_path, language, **kwargs):
        """
        inference with config
        """
        assert (
            language in self.config.languages
        ), f" ❗ Language {language} is not supported. Supported languages are {self.config.languages}"
        # Use generally found best tuning knobs for generation.
        settings = {
            "temperature": config.temperature,
            "length_penalty": config.length_penalty,
            "repetition_penalty": config.repetition_penalty,
            "top_k": config.top_k,
            "top_p": config.top_p,
            "cond_free_k": config.cond_free_k,
            "diffusion_temperature": config.diffusion_temperature,
            "decoder_iterations": config.decoder_iterations,
            "decoder_sampler": config.decoder_sampler,
        }
        settings.update(kwargs)  # allow overriding of preset settings with kwargs
        return self.full_inference(text, ref_audio_path, language, **settings)

    @torch.inference_mode()
    def full_inference(
        self,
        text,
        ref_audio_path,
        language,
        # GPT inference
        temperature=0.65,
        length_penalty=1,
        repetition_penalty=2.0,
        top_k=50,
        top_p=0.85,
        gpt_cond_len=6,
        do_sample=True,
        # Decoder inference
        decoder_iterations=100,
        cond_free=True,
        cond_free_k=2,
        diffusion_temperature=1.0,
        decoder_sampler="ddim",
        decoder="hifigan",
        **hf_generate_kwargs,
    ):
        """
        This function produces an audio clip of the given text being spoken with the given reference voice.

        Args:
            text: (str) Text to be spoken.

            ref_audio_path: (str) Path to a reference audio file to be used for cloning. This audio file should be >3
                seconds long.

            language: (str) Language of the voice to be generated.

            temperature: (float) The softmax temperature of the autoregressive model. Defaults to 0.65.

            length_penalty: (float) A length penalty applied to the autoregressive decoder. Higher settings causes the
                model to produce more terse outputs. Defaults to 1.0.

            repetition_penalty: (float) A penalty that prevents the autoregressive decoder from repeating itself during
                decoding. Can be used to reduce the incidence of long silences or "uhhhhhhs", etc. Defaults to 2.0.

            top_k: (int) K value used in top-k sampling. [0,inf]. Lower values mean the decoder produces more "likely"
                (aka boring) outputs. Defaults to 50.

            top_p: (float) P value used in nucleus sampling. (0,1]. Lower values mean the decoder produces more "likely"
                (aka boring) outputs. Defaults to 0.8.

            gpt_cond_len: (int) Length of the audio used for cloning. If audio is shorter, then audio length is used
                else the first `gpt_cond_len` secs is used. Defaults to 6 seconds.

            decoder_iterations: (int) Number of diffusion steps to perform. [0,4000]. More steps means the network has
                more chances to iteratively refine the output, which should theoretically mean a higher quality output.
                Generally a value above 250 is not noticeably better, however. Defaults to 100.

            cond_free: (bool) Whether or not to perform conditioning-free diffusion. Conditioning-free diffusion
                performs two forward passes for each diffusion step: one with the outputs of the autoregressive model
                and one with no conditioning priors. The output of the two is blended according to the cond_free_k
                value below. Conditioning-free diffusion is the real deal, and dramatically improves realism.
                Defaults to True.

            cond_free_k: (float) Knob that determines how to balance the conditioning free signal with the
                conditioning-present signal. [0,inf]. As cond_free_k increases, the output becomes dominated by the
                conditioning-free signal. Defaults to 2.0.

            diffusion_temperature: (float) Controls the variance of the noise fed into the diffusion model. [0,1].
                Values at 0 re the "mean" prediction of the diffusion network and will sound bland and smeared.
                Defaults to 1.0.

            decoder: (str) Selects the decoder to use between ("hifigan", "ne_hifigan" and "diffusion")
                Defaults to hifigan

            hf_generate_kwargs: (**kwargs) The huggingface Transformers generate API is used for the autoregressive
                transformer. Extra keyword args fed to this function get forwarded directly to that API. Documentation
                here: https://huggingface.co/docs/transformers/internal/generation_utils

        Returns:
            Generated audio clip(s) as a torch tensor. Shape 1,S if k=1 else, (k,1,S) where S is the sample length.
            Sample rate is 24kHz.
        """
        (gpt_cond_latent, diffusion_conditioning, speaker_embedding) = self.get_conditioning_latents(
            audio_path=ref_audio_path, gpt_cond_len=gpt_cond_len
        )
        return self.inference(
            text,
            language,
            gpt_cond_latent,
            speaker_embedding,
            diffusion_conditioning,
            temperature=temperature,
            length_penalty=length_penalty,
            repetition_penalty=repetition_penalty,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            decoder_iterations=decoder_iterations,
            cond_free=cond_free,
            cond_free_k=cond_free_k,
            diffusion_temperature=diffusion_temperature,
            decoder_sampler=decoder_sampler,
            decoder=decoder,
            **hf_generate_kwargs,
        )

    @torch.inference_mode()
    def inference(
        self,
        text,
        language,
        gpt_cond_latent,
        speaker_embedding,
        diffusion_conditioning,
        # GPT inference
        temperature=0.65,
        length_penalty=1,
        repetition_penalty=2.0,
        top_k=50,
        top_p=0.85,
        do_sample=True,
        # Decoder inference
        decoder_iterations=100,
        cond_free=True,
        cond_free_k=2,
        diffusion_temperature=1.0,
        decoder_sampler="ddim",
        decoder="hifigan",
        **hf_generate_kwargs,
    ):
        text = text.strip().lower()
        text_tokens = torch.IntTensor(self.tokenizer.encode(text, lang=language)).unsqueeze(0).to(self.device)

        assert (
            text_tokens.shape[-1] < self.args.gpt_max_text_tokens
        ), " ❗ XTTS can only generate text with a maximum of 400 tokens."

        if not self.args.use_hifigan:
            diffuser = load_discrete_vocoder_diffuser(
                desired_diffusion_steps=decoder_iterations,
                cond_free=cond_free,
                cond_free_k=cond_free_k,
                sampler=decoder_sampler,
            )

        with torch.no_grad():
            gpt_codes = self.gpt.generate(
                cond_latents=gpt_cond_latent,
                text_inputs=text_tokens,
                input_tokens=None,
                do_sample=do_sample,
                top_p=top_p,
                top_k=top_k,
                temperature=temperature,
                num_return_sequences=self.gpt_batch_size,
                length_penalty=length_penalty,
                repetition_penalty=repetition_penalty,
                output_attentions=False,
                **hf_generate_kwargs,
            )
            expected_output_len = torch.tensor(
                [gpt_codes.shape[-1] * self.gpt.code_stride_len], device=text_tokens.device
            )

            text_len = torch.tensor([text_tokens.shape[-1]], device=self.device)
            gpt_latents = self.gpt(
                text_tokens,
                text_len,
                gpt_codes,
                expected_output_len,
                cond_latents=gpt_cond_latent,
                return_attentions=False,
                return_latent=True,
            )
            silence_token = 83
            ctokens = 0
            for k in range(gpt_codes.shape[-1]):
                if gpt_codes[0, k] == silence_token:
                    ctokens += 1
                else:
                    ctokens = 0
                if ctokens > 8:
                    gpt_latents = gpt_latents[:, :k]
                    break

            if decoder == "hifigan":
                assert hasattr(
                    self, "hifigan_decoder"
                ), "You must enable hifigan decoder to use it by setting config `use_hifigan: true`"
                wav = self.hifigan_decoder(gpt_latents, g=speaker_embedding)
            elif decoder == "ne_hifigan":
                assert hasattr(
                    self, "ne_hifigan_decoder"
                ), "You must enable ne_hifigan decoder to use it by setting config `use_ne_hifigan: true`"
                wav = self.ne_hifigan_decoder(gpt_latents, g=speaker_embedding)
            else:
                assert hasattr(
                    self, "diffusion_decoder"
                ), "You must disable hifigan decoders to use difffusion by setting config `use_ne_hifigan: false` and `use_hifigan: false`"
                mel = do_spectrogram_diffusion(
                    self.diffusion_decoder,
                    diffuser,
                    gpt_latents,
                    diffusion_conditioning,
                    temperature=diffusion_temperature,
                )
                wav = self.vocoder.inference(mel)

        return {"wav": wav.cpu().numpy().squeeze()}

    def handle_chunks(self, wav_gen, wav_gen_prev, wav_overlap, overlap_len):
        """Handle chunk formatting in streaming mode"""
        wav_chunk = wav_gen[:-overlap_len]
        if wav_gen_prev is not None:
            wav_chunk = wav_gen[(wav_gen_prev.shape[0] - overlap_len) : -overlap_len]
        if wav_overlap is not None:
            crossfade_wav = wav_chunk[:overlap_len]
            crossfade_wav = crossfade_wav * torch.linspace(0.0, 1.0, overlap_len).to(crossfade_wav.device)
            wav_chunk[:overlap_len] = wav_overlap * torch.linspace(1.0, 0.0, overlap_len).to(wav_overlap.device)
            wav_chunk[:overlap_len] += crossfade_wav
        wav_overlap = wav_gen[-overlap_len:]
        wav_gen_prev = wav_gen
        return wav_chunk, wav_gen_prev, wav_overlap

    @torch.inference_mode()
    def inference_stream(
        self,
        text,
        language,
        gpt_cond_latent,
        speaker_embedding,
        # Streaming
        stream_chunk_size=20,
        overlap_wav_len=1024,
        # GPT inference
        temperature=0.65,
        length_penalty=1,
        repetition_penalty=2.0,
        top_k=50,
        top_p=0.85,
        do_sample=True,
        # Decoder inference
        decoder="hifigan",
        **hf_generate_kwargs,
    ):
        assert hasattr(
            self, "hifigan_decoder"
        ), "`inference_stream` requires use_hifigan to be set to true in the config.model_args, diffusion is too slow to stream."
        text = text.strip().lower()
        text_tokens = torch.IntTensor(self.tokenizer.encode(text, lang=language)).unsqueeze(0).to(self.device)

        fake_inputs = self.gpt.compute_embeddings(
            gpt_cond_latent.to(self.device),
            text_tokens,
        )
        gpt_generator = self.gpt.get_generator(
            fake_inputs=fake_inputs,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            do_sample=do_sample,
            num_beams=1,
            num_return_sequences=1,
            length_penalty=float(length_penalty),
            repetition_penalty=float(repetition_penalty),
            output_attentions=False,
            output_hidden_states=True,
            **hf_generate_kwargs,
        )

        last_tokens = []
        all_latents = []
        wav_gen_prev = None
        wav_overlap = None
        is_end = False

        while not is_end:
            try:
                x, latent = next(gpt_generator)
                last_tokens += [x]
                all_latents += [latent]
            except StopIteration:
                is_end = True

            if is_end or (stream_chunk_size > 0 and len(last_tokens) >= stream_chunk_size):
                gpt_latents = torch.cat(all_latents, dim=0)[None, :]
                if decoder == "hifigan":
                    assert hasattr(
                        self, "hifigan_decoder"
                    ), "You must enable hifigan decoder to use it by setting config `use_hifigan: true`"
                    wav_gen = self.hifigan_decoder(gpt_latents, g=speaker_embedding.to(self.device))
                elif decoder == "ne_hifigan":
                    assert hasattr(
                        self, "ne_hifigan_decoder"
                    ), "You must enable ne_hifigan decoder to use it by setting config `use_ne_hifigan: true`"
                    wav_gen = self.ne_hifigan_decoder(gpt_latents, g=speaker_embedding.to(self.device))
                else:
                    raise NotImplementedError("Diffusion for streaming inference not implemented.")
                wav_chunk, wav_gen_prev, wav_overlap = self.handle_chunks(
                    wav_gen.squeeze(), wav_gen_prev, wav_overlap, overlap_wav_len
                )
                last_tokens = []
                yield wav_chunk

    def forward(self):
        raise NotImplementedError(
            "XTTS has a dedicated trainer, please check the XTTS docs: https://tts.readthedocs.io/en/dev/models/xtts.html#training"
        )

    def eval_step(self):
        raise NotImplementedError(
            "XTTS has a dedicated trainer, please check the XTTS docs: https://tts.readthedocs.io/en/dev/models/xtts.html#training"
        )

    @staticmethod
    def init_from_config(config: "XttsConfig", **kwargs):  # pylint: disable=unused-argument
        return Xtts(config)

    def eval(self):  # pylint: disable=redefined-builtin
        """Sets the model to evaluation mode. Overrides the default eval() method to also set the GPT model to eval mode."""
        self.gpt.init_gpt_for_inference()
        super().eval()

    def get_compatible_checkpoint_state_dict(self, model_path):
        checkpoint = load_fsspec(model_path, map_location=torch.device("cpu"))["model"]
        ignore_keys = ["diffusion_decoder", "vocoder"] if self.args.use_hifigan or self.args.use_ne_hifigan else []
        ignore_keys += [] if self.args.use_hifigan else ["hifigan_decoder"]
        ignore_keys += [] if self.args.use_ne_hifigan else ["ne_hifigan_decoder"]
        # remove xtts gpt trainer extra keys
        ignore_keys += ["torch_mel_spectrogram_style_encoder", "torch_mel_spectrogram_dvae", "dvae"]
        for key in list(checkpoint.keys()):
            # check if it is from the coqui Trainer if so convert it
            if key.startswith("xtts."):
                new_key = key.replace("xtts.", "")
                checkpoint[new_key] = checkpoint[key]
                del checkpoint[key]
                key = new_key

            # remove unused keys
            if key.split(".")[0] in ignore_keys:
                del checkpoint[key]

        return checkpoint

    def load_checkpoint(
        self,
        config,
        checkpoint_dir=None,
        checkpoint_path=None,
        vocab_path=None,
        eval=True,
        strict=True,
        use_deepspeed=False,
    ):
        """
        Loads a checkpoint from disk and initializes the model's state and tokenizer.

        Args:
            config (dict): The configuration dictionary for the model.
            checkpoint_dir (str, optional): The directory where the checkpoint is stored. Defaults to None.
            checkpoint_path (str, optional): The path to the checkpoint file. Defaults to None.
            vocab_path (str, optional): The path to the vocabulary file. Defaults to None.
            eval (bool, optional): Whether to set the model to evaluation mode. Defaults to True.
            strict (bool, optional): Whether to strictly enforce that the keys in the checkpoint match the keys in the model. Defaults to True.

        Returns:
            None
        """

        model_path = checkpoint_path or os.path.join(checkpoint_dir, "model.pth")
        vocab_path = vocab_path or os.path.join(checkpoint_dir, "vocab.json")

        if os.path.exists(vocab_path):
            self.tokenizer = VoiceBpeTokenizer(vocab_file=vocab_path)

        self.init_models()

        checkpoint = self.get_compatible_checkpoint_state_dict(model_path)

        # deal with v1 and v1.1. V1 has the init_gpt_for_inference keys, v1.1 do not
        try:
            self.load_state_dict(checkpoint, strict=strict)
        except:
            if eval:
                self.gpt.init_gpt_for_inference(kv_cache=self.args.kv_cache)
            self.load_state_dict(checkpoint, strict=strict)

        if eval:
            if hasattr(self, "hifigan_decoder"):
                self.hifigan_decoder.eval()
            if hasattr(self, "ne_hifigan_decoder"):
                self.hifigan_decoder.eval()
            if hasattr(self, "diffusion_decoder"):
                self.diffusion_decoder.eval()
            if hasattr(self, "vocoder"):
                self.vocoder.eval()
            self.gpt.init_gpt_for_inference(kv_cache=self.args.kv_cache, use_deepspeed=use_deepspeed)
            self.gpt.eval()

    def train_step(self):
        raise NotImplementedError(
            "XTTS has a dedicated trainer, please check the XTTS docs: https://tts.readthedocs.io/en/dev/models/xtts.html#training"
        )
