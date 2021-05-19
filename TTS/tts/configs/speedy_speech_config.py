from dataclasses import dataclass, field

from TTS.tts.configs.shared_configs import BaseTTSConfig


@dataclass
class SpeedySpeechConfig(BaseTTSConfig):
    """Defines parameters for Speedy Speech (feed-forward encoder-decoder) based models.

    Example:

        >>> from TTS.tts.configs import SpeedySpeechConfig
        >>> config = SpeedySpeechConfig()

    Args:
        model (str):
            Model name used for selecting the right model at initialization. Defaults to `speedy_speech`.
        positional_encoding (bool):
            enable / disable positional encoding applied to the encoder output. Defaults to True.
        hidden_channels (int):
            Base number of hidden channels. Defines all the layers expect ones defined by the specific encoder or decoder
            parameters. Defaults to 128.
        encoder_type (str):
            Type of the encoder used by the model. Look at `TTS.tts.layers.feed_forward.encoder` for more details.
            Defaults to `residual_conv_bn`.
        encoder_params (dict):
            Parameters used to define the encoder network. Look at `TTS.tts.layers.feed_forward.encoder` for more details.
            Defaults to `{"kernel_size": 4, "dilations": [1, 2, 4, 1, 2, 4, 1, 2, 4, 1, 2, 4, 1], "num_conv_blocks": 2, "num_res_blocks": 13}`
        decoder_type (str):
            Type of the decoder used by the model. Look at `TTS.tts.layers.feed_forward.decoder` for more details.
            Defaults to `residual_conv_bn`.
        decoder_params (dict):
            Parameters used to define the decoder network. Look at `TTS.tts.layers.feed_forward.decoder` for more details.
            Defaults to `{"kernel_size": 4, "dilations": [1, 2, 4, 8, 1, 2, 4, 8, 1, 2, 4, 8, 1, 2, 4, 8, 1], "num_conv_blocks": 2, "num_res_blocks": 17}`
        hidden_channels_encoder (int):
            Number of base hidden channels used by the encoder network. It defines the input and the output channel sizes,
            and for some encoder types internal hidden channels sizes too. Defaults to 192.
        hidden_channels_decoder (int):
            Number of base hidden channels used by the decoder WaveNet network. Defaults to 192 as in the original work.
        hidden_channels_duration_predictor (int):
            Number of layer channels of the duration predictor network. Defaults to 256 as in the original work.
        data_dep_init_steps (int):
            Number of steps used for computing normalization parameters at the beginning of the training. GlowTTS uses
            Activation Normalization that pre-computes normalization stats at the beginning and use the same values
            for the rest. Defaults to 10.
        use_speaker_embedding (bool):
            enable / disable using speaker embeddings for multi-speaker models. If set True, the model is
            in the multi-speaker mode. Defaults to False.
        use_external_speaker_embedding_file (bool):
            enable /disable using external speaker embeddings in place of the learned embeddings. Defaults to False.
        external_speaker_embedding_file (str):
            Path to the file including pre-computed speaker embeddings. Defaults to None.
        noam_schedule (bool):
            enable / disable the use of Noam LR scheduler. Defaults to False.
        warmup_steps (int):
            Number of warm-up steps for the Noam scheduler. Defaults 4000.
        lr (float):
            Initial learning rate. Defaults to `1e-3`.
        wd (float):
            Weight decay coefficient. Defaults to `1e-7`.
        ssim_alpha (float):
            Weight for the SSIM loss. If set <= 0, disables the SSIM loss. Defaults to 1.0.
        huber_alpha (float):
            Weight for the duration predictor's loss. Defaults to 1.0.
        l1_alpha (float):
            Weight for the L1 spectrogram loss. If set <= 0, disables the L1 loss. Defaults to 1.0.
        min_seq_len (int):
            Minimum input sequence length to be used at training.
        max_seq_len (int):
            Maximum input sequence length to be used at training. Larger values result in more VRAM usage.
    """

    model: str = "speedy_speech"
    # model specific params
    positional_encoding: bool = True
    hidden_channels: int = 128
    encoder_type: str = "residual_conv_bn"
    encoder_params: dict = field(
        default_factory=lambda: {
            "kernel_size": 4,
            "dilations": [1, 2, 4, 1, 2, 4, 1, 2, 4, 1, 2, 4, 1],
            "num_conv_blocks": 2,
            "num_res_blocks": 13,
        }
    )
    decoder_type: str = "residual_conv_bn"
    decoder_params: dict = field(
        default_factory=lambda: {
            "kernel_size": 4,
            "dilations": [1, 2, 4, 8, 1, 2, 4, 8, 1, 2, 4, 8, 1, 2, 4, 8, 1],
            "num_conv_blocks": 2,
            "num_res_blocks": 17,
        }
    )

    # multi-speaker settings
    use_speaker_embedding: bool = False
    use_external_speaker_embedding_file: bool = False
    external_speaker_embedding_file: str = False

    # optimizer parameters
    noam_schedule: bool = False
    warmup_steps: int = 4000
    lr: float = 1e-4
    wd: float = 1e-6
    grad_clip: float = 5.0

    # loss params
    ssim_alpha: float = 1.0
    huber_alpha: float = 1.0
    l1_alpha: float = 1.0

    # overrides
    min_seq_len: int = 13
    max_seq_len: int = 200
    r: int = 1  # DO NOT CHANGE
