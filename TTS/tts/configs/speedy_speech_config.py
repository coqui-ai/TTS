from dataclasses import dataclass, field
from typing import List

from TTS.tts.configs.shared_configs import BaseTTSConfig
from TTS.tts.models.speedy_speech import SpeedySpeechArgs


@dataclass
class SpeedySpeechConfig(BaseTTSConfig):
    """Defines parameters for Speedy Speech (feed-forward encoder-decoder) based models.

    Example:

        >>> from TTS.tts.configs import SpeedySpeechConfig
        >>> config = SpeedySpeechConfig()

    Args:
        model (str):
            Model name used for selecting the right model at initialization. Defaults to `speedy_speech`.
        model_args (Coqpit):
            Model class arguments. Check `SpeedySpeechArgs` for more details. Defaults to `SpeedySpeechArgs()`.
        data_dep_init_steps (int):
            Number of steps used for computing normalization parameters at the beginning of the training. GlowTTS uses
            Activation Normalization that pre-computes normalization stats at the beginning and use the same values
            for the rest. Defaults to 10.
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
    model_args: SpeedySpeechArgs = field(default_factory=SpeedySpeechArgs)

    # multi-speaker settings
    use_speaker_embedding: bool = False
    use_d_vector_file: bool = False
    d_vector_file: str = False

    # optimizer parameters
    optimizer: str = "RAdam"
    optimizer_params: dict = field(default_factory=lambda: {"betas": [0.9, 0.998], "weight_decay": 1e-6})
    lr_scheduler: str = None
    lr_scheduler_params: dict = None
    lr: float = 1e-4
    grad_clip: float = 5.0

    # loss params
    ssim_alpha: float = 1.0
    huber_alpha: float = 1.0
    l1_alpha: float = 1.0

    # overrides
    min_seq_len: int = 13
    max_seq_len: int = 200
    r: int = 1  # DO NOT CHANGE

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
