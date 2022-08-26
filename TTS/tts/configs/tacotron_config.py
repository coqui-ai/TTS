from dataclasses import dataclass, field
from typing import List

from TTS.tts.configs.shared_configs import BaseTTSConfig, CapacitronVAEConfig, GSTConfig


@dataclass
class TacotronConfig(BaseTTSConfig):
    """Defines parameters for Tacotron based models.

    Example:

        >>> from TTS.tts.configs.tacotron_config import TacotronConfig
        >>> config = TacotronConfig()

    Args:
        model (str):
            Model name used to select the right model class to initilize. Defaults to `Tacotron`.
        use_gst (bool):
            enable / disable the use of Global Style Token modules. Defaults to False.
        gst (GSTConfig):
            Instance of `GSTConfig` class.
        gst_style_input (str):
            Path to the wav file used at inference to set the speech style through GST. If `GST` is enabled and
            this is not defined, the model uses a zero vector as an input. Defaults to None.
        use_capacitron_vae (bool):
            enable / disable the use of Capacitron modules. Defaults to False.
        capacitron_vae (CapacitronConfig):
            Instance of `CapacitronConfig` class.
        num_chars (int):
            Number of characters used by the model. It must be defined before initializing the model. Defaults to None.
        num_speakers (int):
            Number of speakers for multi-speaker models. Defaults to 1.
        r (int):
            Initial number of output frames that the decoder computed per iteration. Larger values makes training and inference
            faster but reduces the quality of the output frames. This must be equal to the largest `r` value used in
            `gradual_training` schedule. Defaults to 1.
        gradual_training (List[List]):
            Parameters for the gradual training schedule. It is in the form `[[a, b, c], [d ,e ,f] ..]` where `a` is
            the step number to start using the rest of the values, `b` is the `r` value and `c` is the batch size.
            If sets None, no gradual training is used. Defaults to None.
        memory_size (int):
            Defines the number of previous frames used by the Prenet. If set to < 0, then it uses only the last frame.
            Defaults to -1.
        prenet_type (str):
            `original` or `bn`. `original` sets the default Prenet and `bn` uses Batch Normalization version of the
            Prenet. Defaults to `original`.
        prenet_dropout (bool):
            enables / disables the use of dropout in the Prenet. Defaults to True.
        prenet_dropout_at_inference (bool):
            enable / disable the use of dropout in the Prenet at the inference time. Defaults to False.
        stopnet (bool):
            enable /disable the Stopnet that predicts the end of the decoder sequence. Defaults to True.
        stopnet_pos_weight (float):
            Weight that is applied to over-weight positive instances in the Stopnet loss. Use larger values with
            datasets with longer sentences. Defaults to 0.2.
        max_decoder_steps (int):
            Max number of steps allowed for the decoder. Defaults to 50.
        encoder_in_features (int):
            Channels of encoder input and character embedding tensors. Defaults to 256.
        decoder_in_features (int):
            Channels of decoder input and encoder output tensors. Defaults to 256.
        out_channels (int):
            Channels of the final model output. It must match the spectragram size. Defaults to 80.
        separate_stopnet (bool):
            Use a distinct Stopnet which is trained separately from the rest of the model. Defaults to True.
        attention_type (str):
            attention type. Check ```TTS.tts.layers.attentions.init_attn```. Defaults to 'original'.
        attention_heads (int):
            Number of attention heads for GMM attention. Defaults to 5.
        windowing (bool):
            It especially useful at inference to keep attention alignment diagonal. Defaults to False.
        use_forward_attn (bool):
            It is only valid if ```attn_type``` is ```original```.  Defaults to False.
        forward_attn_mask (bool):
            enable/disable extra masking over forward attention. It is useful at inference to prevent
            possible attention failures. Defaults to False.
        transition_agent (bool):
            enable/disable transition agent in forward attention. Defaults to False.
        location_attn (bool):
            enable/disable location sensitive attention as in the original Tacotron2 paper.
            It is only valid if ```attn_type``` is ```original```. Defaults to True.
        bidirectional_decoder (bool):
            enable/disable bidirectional decoding. Defaults to False.
        double_decoder_consistency (bool):
            enable/disable double decoder consistency. Defaults to False.
        ddc_r (int):
            reduction rate used by the coarse decoder when `double_decoder_consistency` is in use. Set this
            as a multiple of the `r` value. Defaults to 6.
        speakers_file (str):
            Path to the speaker mapping file for the Speaker Manager. Defaults to None.
        use_speaker_embedding (bool):
            enable / disable using speaker embeddings for multi-speaker models. If set True, the model is
            in the multi-speaker mode. Defaults to False.
        use_d_vector_file (bool):
            enable /disable using external speaker embeddings in place of the learned embeddings. Defaults to False.
        d_vector_file (str):
            Path to the file including pre-computed speaker embeddings. Defaults to None.
        optimizer (str):
            Optimizer used for the training. Set one from `torch.optim.Optimizer` or `TTS.utils.training`.
            Defaults to `RAdam`.
        optimizer_params (dict):
            Optimizer kwargs. Defaults to `{"betas": [0.8, 0.99], "weight_decay": 0.0}`
        lr_scheduler (str):
            Learning rate scheduler for the training. Use one from `torch.optim.Scheduler` schedulers or
            `TTS.utils.training`. Defaults to `NoamLR`.
        lr_scheduler_params (dict):
            Parameters for the generator learning rate scheduler. Defaults to `{"warmup": 4000}`.
        lr (float):
            Initial learning rate. Defaults to `1e-4`.
        wd (float):
            Weight decay coefficient. Defaults to `1e-6`.
        grad_clip (float):
            Gradient clipping threshold. Defaults to `5`.
        seq_len_norm (bool):
            enable / disable the sequnce length normalization in the loss functions. If set True, loss of a sample
            is divided by the sequence length. Defaults to False.
        loss_masking (bool):
            enable / disable masking the paddings of the samples in loss computation. Defaults to True.
        decoder_loss_alpha (float):
            Weight for the decoder loss of the Tacotron model. If set less than or equal to zero, it disables the
            corresponding loss function. Defaults to 0.25
        postnet_loss_alpha (float):
            Weight for the postnet loss of the Tacotron model. If set less than or equal to zero, it disables the
            corresponding loss function. Defaults to 0.25
        postnet_diff_spec_alpha (float):
            Weight for the postnet differential loss of the Tacotron model. If set less than or equal to zero, it disables the
            corresponding loss function. Defaults to 0.25
        decoder_diff_spec_alpha (float):

            Weight for the decoder differential loss of the Tacotron model. If set less than or equal to zero, it disables the
            corresponding loss function. Defaults to 0.25
        decoder_ssim_alpha (float):
            Weight for the decoder SSIM loss of the Tacotron model. If set less than or equal to zero, it disables the
            corresponding loss function. Defaults to 0.25
        postnet_ssim_alpha (float):
            Weight for the postnet SSIM loss of the Tacotron model. If set less than or equal to zero, it disables the
            corresponding loss function. Defaults to 0.25
        ga_alpha (float):
            Weight for the guided attention loss. If set less than or equal to zero, it disables the corresponding loss
            function. Defaults to 5.
    """

    model: str = "tacotron"
    # model_params: TacotronArgs = field(default_factory=lambda: TacotronArgs())
    use_gst: bool = False
    gst: GSTConfig = None
    gst_style_input: str = None

    use_capacitron_vae: bool = False
    capacitron_vae: CapacitronVAEConfig = None

    # model specific params
    num_speakers: int = 1
    num_chars: int = 0
    r: int = 2
    gradual_training: List[List[int]] = None
    memory_size: int = -1
    prenet_type: str = "original"
    prenet_dropout: bool = True
    prenet_dropout_at_inference: bool = False
    stopnet: bool = True
    separate_stopnet: bool = True
    stopnet_pos_weight: float = 0.2
    max_decoder_steps: int = 10000
    encoder_in_features: int = 256
    decoder_in_features: int = 256
    decoder_output_dim: int = 80
    out_channels: int = 513

    # attention layers
    attention_type: str = "original"
    attention_heads: int = None
    attention_norm: str = "sigmoid"
    attention_win: bool = False
    windowing: bool = False
    use_forward_attn: bool = False
    forward_attn_mask: bool = False
    transition_agent: bool = False
    location_attn: bool = True

    # advance methods
    bidirectional_decoder: bool = False
    double_decoder_consistency: bool = False
    ddc_r: int = 6

    # multi-speaker settings
    speakers_file: str = None
    use_speaker_embedding: bool = False
    speaker_embedding_dim: int = 512
    use_d_vector_file: bool = False
    d_vector_file: str = False
    d_vector_dim: int = None

    # optimizer parameters
    optimizer: str = "RAdam"
    optimizer_params: dict = field(default_factory=lambda: {"betas": [0.9, 0.998], "weight_decay": 1e-6})
    lr_scheduler: str = "NoamLR"
    lr_scheduler_params: dict = field(default_factory=lambda: {"warmup_steps": 4000})
    lr: float = 1e-4
    grad_clip: float = 5.0
    seq_len_norm: bool = False
    loss_masking: bool = True

    # loss params
    decoder_loss_alpha: float = 0.25
    postnet_loss_alpha: float = 0.25
    postnet_diff_spec_alpha: float = 0.25
    decoder_diff_spec_alpha: float = 0.25
    decoder_ssim_alpha: float = 0.25
    postnet_ssim_alpha: float = 0.25
    ga_alpha: float = 5.0

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

    def check_values(self):
        if self.gradual_training:
            assert (
                self.gradual_training[0][1] == self.r
            ), f"[!] the first scheduled gradual training `r` must be equal to the model's `r` value. {self.gradual_training[0][1]} vs {self.r}"
        if self.model == "tacotron" and self.audio is not None:
            assert self.out_channels == (
                self.audio.fft_size // 2 + 1
            ), f"{self.out_channels} vs {self.audio.fft_size // 2 + 1}"
        if self.model == "tacotron2" and self.audio is not None:
            assert self.out_channels == self.audio.num_mels
