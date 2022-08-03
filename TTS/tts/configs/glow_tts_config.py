from dataclasses import dataclass, field
from typing import List

from TTS.tts.configs.shared_configs import BaseTTSConfig


@dataclass
class GlowTTSConfig(BaseTTSConfig):
    """Defines parameters for GlowTTS model.

    Example:

        >>> from TTS.tts.configs.glow_tts_config import GlowTTSConfig
        >>> config = GlowTTSConfig()

    Args:
        model(str):
            Model name used for selecting the right model at initialization. Defaults to `glow_tts`.
        encoder_type (str):
            Type of the encoder used by the model. Look at `TTS.tts.layers.glow_tts.encoder` for more details.
            Defaults to `rel_pos_transformers`.
        encoder_params (dict):
            Parameters used to define the encoder network. Look at `TTS.tts.layers.glow_tts.encoder` for more details.
            Defaults to `{"kernel_size": 3, "dropout_p": 0.1, "num_layers": 6, "num_heads": 2, "hidden_channels_ffn": 768}`
        use_encoder_prenet (bool):
            enable / disable the use of a prenet for the encoder. Defaults to True.
        hidden_channels_enc (int):
            Number of base hidden channels used by the encoder network. It defines the input and the output channel sizes,
            and for some encoder types internal hidden channels sizes too. Defaults to 192.
        hidden_channels_dec (int):
            Number of base hidden channels used by the decoder WaveNet network. Defaults to 192 as in the original work.
        hidden_channels_dp (int):
            Number of layer channels of the duration predictor network. Defaults to 256 as in the original work.
        mean_only (bool):
            If true predict only the mean values by the decoder flow. Defaults to True.
        out_channels (int):
            Number of channels of the model output tensor. Defaults to 80.
        num_flow_blocks_dec (int):
            Number of decoder blocks. Defaults to 12.
        inference_noise_scale (float):
            Noise scale used at inference. Defaults to 0.33.
        kernel_size_dec (int):
            Decoder kernel size. Defaults to 5
        dilation_rate (int):
            Rate to increase dilation by each layer in a decoder block. Defaults to 1.
        num_block_layers (int):
            Number of decoder layers in each decoder block.  Defaults to 4.
        dropout_p_dec (float):
            Dropout rate for decoder. Defaults to 0.1.
        num_speaker (int):
            Number of speaker to define the size of speaker embedding layer. Defaults to 0.
        c_in_channels (int):
            Number of speaker embedding channels. It is set to 512 if embeddings are learned. Defaults to 0.
        num_splits (int):
            Number of split levels in inversible conv1x1 operation. Defaults to 4.
        num_squeeze (int):
            Number of squeeze levels. When squeezing channels increases and time steps reduces by the factor
            'num_squeeze'. Defaults to 2.
        sigmoid_scale (bool):
            enable/disable sigmoid scaling in decoder. Defaults to False.
        mean_only (bool):
            If True, encoder only computes mean value and uses constant variance for each time step. Defaults to true.
        encoder_type (str):
            Encoder module type. Possible values are`["rel_pos_transformer", "gated_conv", "residual_conv_bn", "time_depth_separable"]`
            Check `TTS.tts.layers.glow_tts.encoder` for more details. Defaults to `rel_pos_transformers` as in the original paper.
        encoder_params (dict):
            Encoder module parameters. Defaults to None.
        d_vector_dim (int):
            Channels of external speaker embedding vectors. Defaults to 0.
        data_dep_init_steps (int):
            Number of steps used for computing normalization parameters at the beginning of the training. GlowTTS uses
            Activation Normalization that pre-computes normalization stats at the beginning and use the same values
            for the rest. Defaults to 10.
        style_wav_for_test (str):
            Path to the wav file used for changing the style of the speech. Defaults to None.
        inference_noise_scale (float):
            Variance used for sampling the random noise added to the decoder's input at inference. Defaults to 0.0.
        length_scale (float):
            Multiply the predicted durations with this value to change the speech speed. Defaults to 1.
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
            Maximum input sequence length to be used at training. Larger values result in more VRAM usage.
    """

    model: str = "glow_tts"

    # model params
    num_chars: int = None
    encoder_type: str = "rel_pos_transformer"
    encoder_params: dict = field(
        default_factory=lambda: {
            "kernel_size": 3,
            "dropout_p": 0.1,
            "num_layers": 6,
            "num_heads": 2,
            "hidden_channels_ffn": 768,
        }
    )
    use_encoder_prenet: bool = True
    hidden_channels_enc: int = 192
    hidden_channels_dec: int = 192
    hidden_channels_dp: int = 256
    dropout_p_dp: float = 0.1
    dropout_p_dec: float = 0.05
    mean_only: bool = True
    out_channels: int = 80
    num_flow_blocks_dec: int = 12
    inference_noise_scale: float = 0.33
    kernel_size_dec: int = 5
    dilation_rate: int = 1
    num_block_layers: int = 4
    num_speakers: int = 0
    c_in_channels: int = 0
    num_splits: int = 4
    num_squeeze: int = 2
    sigmoid_scale: bool = False
    encoder_type: str = "rel_pos_transformer"
    encoder_params: dict = field(
        default_factory=lambda: {
            "kernel_size": 3,
            "dropout_p": 0.1,
            "num_layers": 6,
            "num_heads": 2,
            "hidden_channels_ffn": 768,
            "input_length": None,
        }
    )
    d_vector_dim: int = 0

    # training params
    data_dep_init_steps: int = 10

    # inference params
    style_wav_for_test: str = None
    inference_noise_scale: float = 0.0
    length_scale: float = 1.0

    # multi-speaker settings
    use_speaker_embedding: bool = False
    speakers_file: str = None
    use_d_vector_file: bool = False
    d_vector_file: str = False

    # optimizer parameters
    optimizer: str = "RAdam"
    optimizer_params: dict = field(default_factory=lambda: {"betas": [0.9, 0.998], "weight_decay": 1e-6})
    lr_scheduler: str = "NoamLR"
    lr_scheduler_params: dict = field(default_factory=lambda: {"warmup_steps": 4000})
    grad_clip: float = 5.0
    lr: float = 1e-3

    # overrides
    min_seq_len: int = 3
    max_seq_len: int = 500
    r: int = 1  # DO NOT CHANGE - TODO: make this immutable once coqpit implements it.

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
