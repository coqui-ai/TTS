from dataclasses import dataclass

from TTS.tts.configs.tacotron_config import TacotronConfig


@dataclass
class Tacotron2Config(TacotronConfig):
    """Defines parameters for Tacotron2 based models.

    Example:

        >>> from TTS.tts.configs import Tacotron2Config
        >>> config = Tacotron2Config()

    Args:
        model (str):
            Model name used to select the right model class to initilize. Defaults to `Tacotron2`.
        use_gst (bool):
            enable / disable the use of Global Style Token modules. Defaults to False.
        gst (GSTConfig):
            Instance of `GSTConfig` class.
        gst_style_input (str):
            Path to the wav file used at inference to set the speech style through GST. If `GST` is enabled and
            this is not defined, the model uses a zero vector as an input. Defaults to None.
        r (int):
            Number of output frames that the decoder computed per iteration. Larger values makes training and inference
            faster but reduces the quality of the output frames. This needs to be tuned considering your own needs.
            Defaults to 1.
        gradual_trainin (List[List]):
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
            datasets with longer sentences. Defaults to 10.
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
            Initial learning rate. Defaults to `1e-4`.
        wd (float):
            Weight decay coefficient. Defaults to `1e-6`.
        grad_clip (float):
            Gradient clipping threshold. Defaults to `5`.
        seq_len_notm (bool):
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

    model: str = "tacotron2"
