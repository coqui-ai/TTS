from dataclasses import dataclass, field

from TTS.tts.configs.shared_configs import BaseTTSConfig


@dataclass
class GlowTTSConfig(BaseTTSConfig):
    """Defines parameters for GlowTTS model.

     Example:

        >>> from TTS.tts.configs import GlowTTSConfig
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
        style_wav_for_test (str):
            Path to the wav file used for changing the style of the speech. Defaults to None.
        inference_noise_scale (float):
            Variance used for sampling the random noise added to the decoder's input at inference. Defaults to 0.0.
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
        min_seq_len (int):
            Minimum input sequence length to be used at training.
        max_seq_len (int):
            Maximum input sequence length to be used at training. Larger values result in more VRAM usage.
    """

    model: str = "glow_tts"

    # model params
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
    hidden_channels_encoder: int = 192
    hidden_channels_decoder: int = 192
    hidden_channels_duration_predictor: int = 256

    # training params
    data_dep_init_steps: int = 10

    # inference params
    style_wav_for_test: str = None
    inference_noise_scale: float = 0.0

    # multi-speaker settings
    use_speaker_embedding: bool = False
    use_external_speaker_embedding_file: bool = False
    external_speaker_embedding_file: str = False

    # optimizer params
    noam_schedule: bool = True
    warmup_steps: int = 4000
    grad_clip: float = 5.0
    lr: float = 1e-3
    wd: float = 0.000001

    # overrides
    min_seq_len: int = 3
    max_seq_len: int = 500
    r: int = 1  # DO NOT CHANGE - TODO: make this immutable once coqpit implements it.
