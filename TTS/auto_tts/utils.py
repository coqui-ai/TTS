from TTS.config.shared_configs import BaseAudioConfig, BaseDatasetConfig


def data_loader(name, path, stats_path=None):
    if name == "ljspeech":
        dataset = BaseDatasetConfig(name="ljspeech", meta_file_train="metadata.csv", path=path)
        audio = BaseAudioConfig(
            ref_level_db=0, trim_db=60, mel_fmin=50.0, mel_fmax=7600.0, spec_gain=1, stats_path=stats_path
        )

    elif name == "vctk":
        dataset = BaseDatasetConfig(
            name="vctk",
            meta_file_train=["p225", "p234", "p238", "p245", "p248", "p261", "p294", "p302", "p326", "p335", "p347"],
            meta_file_val=None,
            path=path,
        )
        audio = BaseAudioConfig(
            sample_rate=22050,
            preemphasis=0.98,
            ref_level_db=20,
            clip_norm=True,
            mel_fmin=0.0,
            mel_fmax=8000.0,
            spec_gain=20,
            do_trim_silence=False,
            trim_db=60,
            power=1.5,
            num_mels=80,
            resample=True,
        )

    elif name == "libri_tts":
        dataset = BaseDatasetConfig(name="libri_tts", meta_file_train=None, meta_file_val=None, path=path)
        audio = BaseAudioConfig(
            resample=False,
            sample_rate=24000,
            preemphasis=0.98,
            ref_level_db=20,
            power=1.5,
            signal_norm=True,
            symmetric_norm=True,
            max_norm=4.0,
            clip_norm=True,
            mel_fmax=8000.0,
            spec_gain=20,
            do_trim_silence=False,
            trim_db=25,
        )
    elif name == "baker":
        dataset = BaseDatasetConfig(name=name, meta_file_train="metadata.csv", meta_file_val=None, path=path)
        audio = BaseAudioConfig(
            sample_rate=22050,
            preemphasis=0.0,
            ref_level_db=0,
            do_trim_silence=True,
            trim_db=60,
            mel_fmin=50.0,
            mel_fmax=7600.0,
            spec_gain=1,
            signal_norm=True,
            symmetric_norm=True,
            clip_norm=True,
            stats_path=stats_path,
        )
    return audio, dataset


def custom_data_loader(sr, audio_path):
    dataset = BaseDatasetConfig(name="ljspeech", meta_file_train="metadata.csv", path=audio_path)
    pass
    # this is for loading custom dataloader, it still takes the ljspeech format but the audio configs will differ
    # with each users data so im thinking of a way to have users define their own audio params with this


def pick_glowtts_encoder(encoder_name: str):
    if encoder_name == "transformer":
        encoder_type = "rel_pos_transformer"
    elif encoder_name == "gated":
        encoder_type = "gated_conv"
    elif encoder_name == "residual_bn":
        encoder_type = "residual_conv_bn"
    elif encoder_name == "time_depth":
        encoder_type = "time_depth_separable"
    else:
        encoder_type = "rel_pos_transformer"
    return encoder_type


def pick_forwardtts_encoder(encoder_name: str):
    if encoder_name == "residual_bn":
        encoder = "residual_conv_bn"
    elif encoder_name == "fftransformer":
        encoder = encoder_name
    elif encoder_name == "position transformer":
        encoder = "relative_position_transformer"
    else:
        print("please select an actual encoder. either residual_bn, fftransformer, or position transformer")
    return encoder


def pick_forwardtts_decoder(decoder_name: str):
    if decoder_name  == "position transformer":
        decoder = "relative_position_transformer"
    elif decoder_name == " residual_bn":
        decoder = "residual_conv_bn"
    elif decoder_name == "wavenet":
        decoder = decoder_name
    elif decoder_name == "fftransformer":
        decoder = decoder_name
    else:
        print("please select either position transformer, residual_bn, wavenet, or fftransformer")
    return decoder
