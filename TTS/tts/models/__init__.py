from TTS.utils.generic_utils import find_module


def setup_model(num_chars, num_speakers, c, speaker_embedding_dim=None):
    print(" > Using model: {}".format(c.model))
    MyModel = find_module("TTS.tts.models", c.model.lower())
    if c.model.lower() in "tacotron":
        model = MyModel(
            num_chars=num_chars + getattr(c, "add_blank", False),
            num_speakers=num_speakers,
            r=c.r,
            postnet_output_dim=int(c.audio["fft_size"] / 2 + 1),
            decoder_output_dim=c.audio["num_mels"],
            use_gst=c.use_gst,
            gst=c.gst,
            memory_size=c.memory_size,
            attn_type=c.attention_type,
            attn_win=c.windowing,
            attn_norm=c.attention_norm,
            prenet_type=c.prenet_type,
            prenet_dropout=c.prenet_dropout,
            prenet_dropout_at_inference=c.prenet_dropout_at_inference,
            forward_attn=c.use_forward_attn,
            trans_agent=c.transition_agent,
            forward_attn_mask=c.forward_attn_mask,
            location_attn=c.location_attn,
            attn_K=c.attention_heads,
            separate_stopnet=c.separate_stopnet,
            bidirectional_decoder=c.bidirectional_decoder,
            double_decoder_consistency=c.double_decoder_consistency,
            ddc_r=c.ddc_r,
            speaker_embedding_dim=speaker_embedding_dim,
        )
    elif c.model.lower() == "tacotron2":
        model = MyModel(
            num_chars=num_chars + getattr(c, "add_blank", False),
            num_speakers=num_speakers,
            r=c.r,
            postnet_output_dim=c.audio["num_mels"],
            decoder_output_dim=c.audio["num_mels"],
            use_gst=c.use_gst,
            gst=c.gst,
            attn_type=c.attention_type,
            attn_win=c.windowing,
            attn_norm=c.attention_norm,
            prenet_type=c.prenet_type,
            prenet_dropout=c.prenet_dropout,
            prenet_dropout_at_inference=c.prenet_dropout_at_inference,
            forward_attn=c.use_forward_attn,
            trans_agent=c.transition_agent,
            forward_attn_mask=c.forward_attn_mask,
            location_attn=c.location_attn,
            attn_K=c.attention_heads,
            separate_stopnet=c.separate_stopnet,
            bidirectional_decoder=c.bidirectional_decoder,
            double_decoder_consistency=c.double_decoder_consistency,
            ddc_r=c.ddc_r,
            speaker_embedding_dim=speaker_embedding_dim,
        )
    elif c.model.lower() == "glow_tts":
        model = MyModel(
            num_chars=num_chars + getattr(c, "add_blank", False),
            hidden_channels_enc=c["hidden_channels_encoder"],
            hidden_channels_dec=c["hidden_channels_decoder"],
            hidden_channels_dp=c["hidden_channels_duration_predictor"],
            out_channels=c.audio["num_mels"],
            encoder_type=c.encoder_type,
            encoder_params=c.encoder_params,
            use_encoder_prenet=c["use_encoder_prenet"],
            inference_noise_scale=c.inference_noise_scale,
            num_flow_blocks_dec=12,
            kernel_size_dec=5,
            dilation_rate=1,
            num_block_layers=4,
            dropout_p_dec=0.05,
            num_speakers=num_speakers,
            c_in_channels=0,
            num_splits=4,
            num_squeeze=2,
            sigmoid_scale=False,
            mean_only=True,
            speaker_embedding_dim=speaker_embedding_dim,
        )
    elif c.model.lower() == "speedy_speech":
        model = MyModel(
            num_chars=num_chars + getattr(c, "add_blank", False),
            out_channels=c.audio["num_mels"],
            hidden_channels=c["hidden_channels"],
            positional_encoding=c["positional_encoding"],
            encoder_type=c["encoder_type"],
            encoder_params=c["encoder_params"],
            decoder_type=c["decoder_type"],
            decoder_params=c["decoder_params"],
            c_in_channels=0,
        )
    elif c.model.lower() == "align_tts":
        model = MyModel(
            num_chars=num_chars + getattr(c, "add_blank", False),
            out_channels=c.audio["num_mels"],
            hidden_channels=c["hidden_channels"],
            hidden_channels_dp=c["hidden_channels_dp"],
            encoder_type=c["encoder_type"],
            encoder_params=c["encoder_params"],
            decoder_type=c["decoder_type"],
            decoder_params=c["decoder_params"],
            c_in_channels=0,
        )
    return model
