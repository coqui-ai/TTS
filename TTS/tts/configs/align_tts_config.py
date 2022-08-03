from dataclasses import dataclass, field
from typing import List

from TTS.tts.configs.shared_configs import BaseTTSConfig
from TTS.tts.models.align_tts import AlignTTSArgs


@dataclass
class AlignTTSConfig(BaseTTSConfig):
    """Defines parameters for AlignTTS model.
    Example:

        >>> from TTS.tts.configs.align_tts_config import AlignTTSConfig
        >>> config = AlignTTSConfig()

    Args:
        model(str):
            Model name used for selecting the right model at initialization. Defaults to `align_tts`.
        positional_encoding (bool):
            enable / disable positional encoding applied to the encoder output. Defaults to True.
        hidden_channels (int):
            Base number of hidden channels. Defines all the layers expect ones defined by the specific encoder or decoder
            parameters. Defaults to 256.
        hidden_channels_dp (int):
            Number of hidden channels of the duration predictor's layers. Defaults to 256.
        encoder_type (str):
            Type of the encoder used by the model. Look at `TTS.tts.layers.feed_forward.encoder` for more details.
            Defaults to `fftransformer`.
        encoder_params (dict):
            Parameters used to define the encoder network. Look at `TTS.tts.layers.feed_forward.encoder` for more details.
            Defaults to `{"hidden_channels_ffn": 1024, "num_heads": 2, "num_layers": 6, "dropout_p": 0.1}`.
        decoder_type (str):
            Type of the decoder used by the model. Look at `TTS.tts.layers.feed_forward.decoder` for more details.
            Defaults to `fftransformer`.
        decoder_params (dict):
            Parameters used to define the decoder network. Look at `TTS.tts.layers.feed_forward.decoder` for more details.
            Defaults to `{"hidden_channels_ffn": 1024, "num_heads": 2, "num_layers": 6, "dropout_p": 0.1}`.
        phase_start_steps (List[int]):
            A list of number of steps required to start the next training phase. AlignTTS has 4 different training
            phases. Thus you need to define 4 different values to enable phase based training. If None, it
            trains the whole model together. Defaults to None.
        ssim_alpha (float):
            Weight for the SSIM loss. If set <= 0, disables the SSIM loss. Defaults to 1.0.
        duration_loss_alpha (float):
            Weight for the duration predictor's loss. Defaults to 1.0.
        mdn_alpha (float):
            Weight for the MDN loss. Defaults to 1.0.
        spec_loss_alpha (float):
            Weight for the MSE spectrogram loss. If set <= 0, disables the L1 loss. Defaults to 1.0.
        use_speaker_embedding (bool):
            enable / disable using speaker embeddings for multi-speaker models. If set True, the model is
            in the multi-speaker mode. Defaults to False.
        use_d_vector_file (bool):
            enable /disable using external speaker embeddings in place of the learned embeddings. Defaults to False.
        d_vector_file (str):
            Path to the file including pre-computed speaker embeddings. Defaults to None.
        noam_schedule (bool):
            enable / disable the use of Noam LR scheduler. Defaults to False.
        warmup_steps (int):
            Number of warm-up steps for the Noam scheduler. Defaults 4000.
        lr (float):
            Initial learning rate. Defaults to `1e-3`.
        wd (float):
            Weight decay coefficient. Defaults to `1e-7`.
        min_seq_len (int):
            Minimum input sequence length to be used at training.
        max_seq_len (int):
            Maximum input sequence length to be used at training. Larger values result in more VRAM usage."""

    model: str = "align_tts"
    # model specific params
    model_args: AlignTTSArgs = field(default_factory=AlignTTSArgs)
    phase_start_steps: List[int] = None

    ssim_alpha: float = 1.0
    spec_loss_alpha: float = 1.0
    dur_loss_alpha: float = 1.0
    mdn_alpha: float = 1.0

    # multi-speaker settings
    use_speaker_embedding: bool = False
    use_d_vector_file: bool = False
    d_vector_file: str = False

    # optimizer parameters
    optimizer: str = "Adam"
    optimizer_params: dict = field(default_factory=lambda: {"betas": [0.9, 0.998], "weight_decay": 1e-6})
    lr_scheduler: str = None
    lr_scheduler_params: dict = None
    lr: float = 1e-4
    grad_clip: float = 5.0

    # overrides
    min_seq_len: int = 13
    max_seq_len: int = 200
    r: int = 1

    # testing
    test_sentences: List[str] = field(
        default_factory=lambda: [
            "It took me quite a long time to develop a voice, and now that I have it I'm not going to be silent.",
            "Be a voice, not an echo.",
            "I'm sorry Dave. I'm afraid I can't do that.",
            "This cake is great. It's so delicious and moist.",
            "Prior to November 22, 1963.",
        ]
    )
