from dataclasses import dataclass, field
from typing import List

from TTS.tts.configs.shared_configs import BaseTTSConfig


@dataclass
class OverFlowConfig(BaseTTSConfig):
    """
    Define parameters for OverFlow model.

    Args:
        BaseTTSConfig (_type_): _description_
    """

    model: str = "overflow"

    # Training and Checkpoint configs
    run_eval_steps: int = 200
    save_step: int = 500
    plot_step: int = 1
    model_param_stats: bool = False

    # data parameters
    force_generate_statistics: bool = False
    mel_statistics_parameter_path: str = None

    # Encoder parameters
    num_chars: int = None
    state_per_phone: int = 2
    encoder_in_out_features: int = 512
    encoder_n_convolutions: int = 3

    # HMM parameters
    out_channels: int = 80
    ar_order: int = 1
    sampling_temp: float = 0.334
    deterministic_transition: bool = True
    duration_threshold: float = 0.55
    use_grad_checkpointing: bool = True
    max_sampling_time: int = 1000

    ## Prenet parameters
    prenet_type: str = "original"
    prenet_dim: int = 256
    prenet_n_layers: int = 2
    prenet_dropout: float = 0.5
    prenet_dropout_at_inference: bool = False
    memory_rnn_dim: int = 1024

    ## Outputnet parameters
    outputnet_size: List[int] = field(default_factory=lambda: [256, 256])
    flat_start_params: dict = field(default_factory=lambda: {"mean": 0.0, "std": 1.0, "transition_p": 0.14})
    std_floor: float = 0.01

    # Decoder parameters
    hidden_channels_dec: int = 150
    kernel_size_dec: int = 5
    dilation_rate: int = 1
    num_flow_blocks_dec: int = 12
    num_block_layers: int = 4
    dropout_p_dec: float = 0.05
    num_splits: int = 4
    num_squeeze: int = 2
    sigmoid_scale: bool = False
    c_in_channels: int = 0

    # optimizer parameters
    optimizer: str = "Adam"
    optimizer_params: dict = field(default_factory=lambda: {"weight_decay": 1e-6})
    grad_clip: float = 40000.0
    lr: float = 1e-3
    lr_scheduler: str = None

    # overrides
    min_seq_len: int = 3
    max_seq_len: int = 500

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

    # Extra needed config
    r: int = 1
    use_d_vector_file: bool = False
    use_speaker_embedding: bool = False

    def check_values(self):
        """Validate the hyperparameters.

        Raises:
            AssertionError: when the parameters network is not defined
            AssertionError: transition probability is not between 0 and 1
        """
        assert self.ar_order > 0, "AR order must be greater than 0 it is an autoregressive model."
        assert (
            self.parameternetwork >= 1
        ), f"Parameter Network must have atleast one layer check the config file for parameter network. Provided: {self.parameternetwork}"
        assert (
            0 < self.flat_start_params["transition_p"] < 1
        ), f"Transition probability must be between 0 and 1. Provided: {self.flat_start_params['transition_p']}"
