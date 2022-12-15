from dataclasses import dataclass, field
from typing import List

from TTS.tts.configs.shared_configs import BaseTTSConfig


@dataclass
class OverflowConfig(BaseTTSConfig):  # The classname has to be camel case
    """
    Define parameters for OverFlow model.

    Example:

        >>> from TTS.tts.configs.overflow_config import OverflowConfig
        >>> config = OverflowConfig()

    Args:
        model (str):
            Model name used to select the right model class to initilize. Defaults to `Overflow`.
        run_eval_steps (int):
            Run evalulation epoch after N steps. If None, waits until training epoch is completed. Defaults to None.
        save_step (int):
            Save local checkpoint every save_step steps. Defaults to 500.
        plot_step (int):
            Plot training stats on the logger every plot_step steps. Defaults to 1.
        model_param_stats (bool):
            Log model parameters stats on the logger dashboard. Defaults to False.
        force_generate_statistics (bool):
            Force generate mel normalization statistics. Defaults to False.
        mel_statistics_parameter_path (str):
            Path to the mel normalization statistics.If the model doesn't finds a file there it will generate statistics.
            Defaults to None.
        num_chars (int):
            Number of characters used by the model. It must be defined before initializing the model. Defaults to None.
        state_per_phone (int):
            Generates N states per phone. Similar, to `add_blank` parameter in GlowTTS but in Overflow it is upsampled by model's encoder. Defaults to 2.
        encoder_in_out_features (int):
            Channels of encoder input and character embedding tensors. Defaults to 512.
        encoder_n_convolutions (int):
            Number of convolution layers in the encoder. Defaults to 3.
        out_channels (int):
            Channels of the final model output. It must match the spectragram size. Defaults to 80.
        ar_order (int):
            Autoregressive order of the model. Defaults to 1. In ablations of Neural HMM it was found that more autoregression while giving more variation hurts naturalness of the synthesised audio.
        sampling_temp (float):
            Variation added to the sample from the latent space of neural HMM. Defaults to 0.334.
        deterministic_transition (bool):
            deterministic duration generation based on duration quantiles as defiend in "S. Ronanki, O. Watts, S. King, and G. E. Henter, “Medianbased generation of synthetic speech durations using a nonparametric approach,” in Proc. SLT, 2016.". Defaults to True.
        duration_threshold (float):
            Threshold for duration quantiles. Defaults to 0.55. Tune this to change the speaking rate of the synthesis, where lower values defines a slower speaking rate and higher values defines a faster speaking rate.
        use_grad_checkpointing (bool):
            Use gradient checkpointing to save memory. In a multi-GPU setting currently pytorch does not supports gradient checkpoint inside a loop so we will have to turn it off then.Adjust depending on whatever get more batch size either by using a single GPU or multi-GPU. Defaults to True.
        max_sampling_time (int):
            Maximum sampling time while synthesising latents from neural HMM. Defaults to 1000.
        prenet_type (str):
            `original` or `bn`. `original` sets the default Prenet and `bn` uses Batch Normalization version of the
            Prenet. Defaults to `original`.
        prenet_dim (int):
            Dimension of the Prenet. Defaults to 256.
        prenet_n_layers (int):
            Number of layers in the Prenet. Defaults to 2.
        prenet_dropout (float):
            Dropout rate of the Prenet. Defaults to 0.5.
        prenet_dropout_at_inference (bool):
            Use dropout at inference time. Defaults to False.
        memory_rnn_dim (int):
            Dimension of the memory LSTM to process the prenet output. Defaults to 1024.
        outputnet_size (list[int]):
            Size of the output network inside the neural HMM. Defaults to [1024].
        flat_start_params (dict):
            Parameters for the flat start initialization of the neural HMM. Defaults to `{"mean": 0.0, "std": 1.0, "transition_p": 0.14}`.
            It will be recomputed when you pass the dataset.
        std_floor (float):
            Floor value for the standard deviation of the neural HMM. Prevents model cheating by putting point mass and getting infinite likelihood at any datapoint. Defaults to 0.01.
            It is called `variance flooring` in standard HMM literature.
        hidden_channels_dec (int):
            Number of base hidden channels used by the decoder WaveNet network. Defaults to 150.
        kernel_size_dec (int):
            Decoder kernel size. Defaults to 5
        dilation_rate (int):
            Rate to increase dilation by each layer in a decoder block. Defaults to 1.
        num_flow_blocks_dec (int):
            Number of decoder layers in each decoder block.  Defaults to 4.
        dropout_p_dec (float):
            Dropout rate of the decoder. Defaults to 0.05.
        num_splits (int):
            Number of split levels in inversible conv1x1 operation. Defaults to 4.
        num_squeeze (int):
            Number of squeeze levels. When squeezing channels increases and time steps reduces by the factor
            'num_squeeze'. Defaults to 2.
        sigmoid_scale (bool):
            enable/disable sigmoid scaling in decoder. Defaults to False.
        c_in_channels (int):
            Unused parameter from GlowTTS's decoder. Defaults to 0.
        optimizer (str):
            Optimizer to use for training. Defaults to `adam`.
        optimizer_params (dict):
            Parameters for the optimizer. Defaults to `{"weight_decay": 1e-6}`.
        grad_clip (float):
            Gradient clipping threshold. Defaults to 40_000.
        lr (float):
            Learning rate. Defaults to 1e-3.
        lr_scheduler (str):
            Learning rate scheduler for the training. Use one from `torch.optim.Scheduler` schedulers or
            `TTS.utils.training`. Defaults to `None`.
        min_seq_len (int):
            Minimum input sequence length to be used at training.
        max_seq_len (int):
            Maximum input sequence length to be used at training. Larger values result in more VRAM usage.
    """

    model: str = "Overflow"

    # Training and Checkpoint configs
    run_eval_steps: int = 100
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
    outputnet_size: List[int] = field(default_factory=lambda: [1024])
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
    min_text_len: int = 10
    max_text_len: int = 500
    min_audio_len: int = 512

    # testing
    test_sentences: List[str] = field(
        default_factory=lambda: [
            "Be a voice, not an echo.",
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
            len(self.outputnet_size) >= 1
        ), f"Parameter Network must have atleast one layer check the config file for parameter network. Provided: {self.parameternetwork}"
        assert (
            0 < self.flat_start_params["transition_p"] < 1
        ), f"Transition probability must be between 0 and 1. Provided: {self.flat_start_params['transition_p']}"
