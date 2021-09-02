import os

from TTS.tts.configs.glow_tts_config import GlowTTSConfig
from TTS.tts.configs.speedy_speech_config import SpeedySpeechArgs, SpeedySpeechConfig
from TTS.tts.configs.tacotron2_config import Tacotron2Config
from TTS.tts.configs.vits_config import VitsConfig
from TTS.vocoder.configs.hifigan_config import HifiganConfig
from TTS.vocoder.configs.multiband_melgan_config import MultibandMelganConfig
from TTS.vocoder.configs.univnet_config import UnivnetConfig
from TTS.vocoder.configs.wavegrad_config import WavegradConfig


class TtsModels:
    """
    This is a class that holds all the model configs.
    If you want to add a new recipe this is where you would add your model config
    made this so the complete recipes file doesnt look so cluttered. If you would like
    to contribute a model for a recipe look at the usage on how you should format the function.

    usage:
            def model_name:
                config = model_config(...)
                return config

    """


    def __init__(
        self, batch_size, mixed_precision, learning_rate, epochs, output_path=os.path.dirname(os.path.abspath(__file__))
    ):

        self.batch_size = batch_size
        self.output_path = output_path
        self.mixed_precision = mixed_precision
        self.learning_rate = learning_rate
        self.epochs = epochs

    def single_speaker_tacotron2_base(
        self, audio, dataset, dla=0.25, pla=0.25, ga=5.0, forward_attn=True, location_attn=True
    ):
        config = Tacotron2Config(
            run_name="single_speaker_taoctron2",
            audio=audio,
            batch_size=self.batch_size,
            eval_batch_size=int(self.batch_size / 2),
            r=2,
            grad_clip=1,
            lr=self.learning_rate,
            decoder_loss_alpha=dla,
            postnet_loss_alpha=pla,
            postnet_diff_spec_alpha=0.25,
            decoder_diff_spec_alpha=0.25,
            decoder_ssim_alpha=0.25,
            postnet_ssim_alpha=0.25,
            ga_alpha=ga,
            stopnet_pos_weight=15.0,
            memory_size=-1,
            prenet_type="original",
            prenet_dropout=True,
            attention_type="original",
            attention_heads=5,
            attention_norm="sigmoid",
            windowing=False,
            use_forward_attn=forward_attn,
            forward_attn_mask=False,
            transition_agent=False,
            location_attn=location_attn,
            stopnet=True,
            separate_stopnet=True,
            print_step=25,
            save_step=10000,
            checkpoint=True,
            text_cleaner="basic_cleaners",
            num_loader_workers=6,
            num_eval_loader_workers=6,
            min_seq_len=6,
            max_seq_len=150,
            output_path=self.output_path,
            datasets=[dataset],
        )
        return config

    def single_speaker_tacotron2_DDC(
        self, audio, dataset, dla=0.25, pla=0.25, ga=5.0, forward_attn=False, location_attn=True
    ):
        config = Tacotron2Config(
            audio=audio,
            run_name="ljspeech-ddc",
            run_description="tacotron2 with double decoder consistency.",
            batch_size=self.batch_size,
            eval_batch_size=self.batch_size // 2,
            mixed_precision=False,
            loss_masking=True,
            decoder_loss_alpha=dla,
            postnet_loss_alpha=pla,
            postnet_diff_spec_alpha=0.25,
            decoder_diff_spec_alpha=0.25,
            decoder_ssim_alpha=0.25,
            postnet_ssim_alpha=0.25,
            ga_alpha=ga,
            stopnet_pos_weight=15.0,
            run_eval=True,
            test_delay_epochs=10,
            max_decoder_steps=1000,
            grad_clip=0.05,
            epochs=self.epochs,
            lr=self.learning_rate,
            memory_size=-1,
            prenet_type="original",
            use_forward_attn=forward_attn,
            prenet_dropout=True,
            attention_type="original",
            location_attn=location_attn,
            double_decoder_consistency=True,
            ddc_r=6,
            attention_norm="sigmoid",
            r=6,
            gradual_training=[
                [0, 6, self.batch_size],
                [10000, 4, self.batch_size // 2],
                [50000, 3, self.batch_size // 2],
                [100000, 2, self.batch_size // 2],
            ],
            stopnet=True,
            separate_stopnet=True,
            print_step=25,
            print_eval=False,
            plot_step=100,
            save_step=10000,
            checkpoint=True,
            text_cleaner="phoneme_cleaners",
            num_loader_workers=4,
            num_eval_loader_workers=4,
            batch_group_size=4,
            min_seq_len=6,
            max_seq_len=180,
            compute_input_seq_cache=True,
            phoneme_cache_path=os.path.join(self.output_path, "phoneme_cache"),
            output_path=self.output_path,
            use_phonemes=False,
            phoneme_language="en-us",
            datasets=[dataset],
        )
        return config

    def single_speaker_tacotron2_DCA(
        self, audio, dataset, dla=0.25, pla=0.25, ga=5.0, forward_attn=False, location_attn=True
    ):
        """This is a tacotron2 dca config for the ljspeech dataset,
        based off the already existing recipe config."""
        config = Tacotron2Config(
            audio=audio,
            run_name="ljspeech-dca",
            run_description="tacotron2 with dynamic conv attention.",
            batch_size=self.batch_size,
            eval_batch_size=self.batch_size // 2,
            mixed_precision=True,
            loss_masking=True,
            decoder_loss_alpha=dla,
            postnet_loss_alpha=pla,
            postnet_diff_spec_alpha=0.25,
            decoder_diff_spec_alpha=0.25,
            decoder_ssim_alpha=0.25,
            postnet_ssim_alpha=0.25,
            ga_alpha=ga,
            stopnet_pos_weight=15.0,
            run_eval=True,
            test_delay_epochs=10,
            max_decoder_steps=1000,
            grad_clip=0.05,
            epochs=self.epochs,
            lr=self.learning_rate,
            memory_size=-1,
            prenet_type="original",
            use_forward_attn=forward_attn,
            prenet_dropout=True,
            attention_type="dynamic_convolution",
            location_attn=location_attn,
            attention_norm="sigmoid",
            r=2,
            stopnet=True,
            separate_stopnet=True,
            print_step=25,
            plot_step=100,
            print_eval=False,
            save_step=10000,
            checkpoint=True,
            text_cleaner="phoneme_cleaners",
            num_loader_workers=4,
            num_eval_loader_workers=4,
            batch_group_size=4,
            min_seq_len=6,
            max_seq_len=180,
            compute_input_seq_cache=True,
            output_path=self.output_path,
            phoneme_cache_path=os.path.join(self.output_path, "phoneme_cache"),
            use_phonemes=False,
            phoneme_language="en-us",
            datasets=[dataset],
        )
        return config

    def ljspeech_glow_tts(self, audio, dataset, encoder):
        if encoder == "transformer":
            encoder_type = "rel_pos_transformer"
        elif encoder == "gated":
            encoder_type = "gated_conv"
        elif encoder == "residual_bn":
            encoder_type = "residual_conv_bn"
        elif encoder == "time_depth":
            encoder_type = "time_depth_separable"
        elif encoder is None:
            encoder_type = "rel_pos_transformer"

        config = GlowTTSConfig(
            audio=audio,
            batch_size=self.batch_size,
            eval_batch_size=self.batch_size // 2,
            num_loader_workers=4,
            num_eval_loader_workers=4,
            run_eval=True,
            test_delay_epochs=-1,
            epochs=self.epochs,
            text_cleaner="english_cleaners",
            use_phonemes=False,
            phoneme_language="en-us",
            phoneme_cache_path=os.path.join(self.output_path, "phoneme_cache"),
            print_step=25,
            print_eval=True,
            mixed_precision=False,
            add_blank=False,
            hidden_channels_enc=192,
            hidden_channels_dec=192,
            hidden_channels_dp=256,
            use_encoder_prenet=True,
            encoder_type=encoder_type,
            r=1,
            loss_masking=True,
            grad_clip=5.0,
            save_step=5000,
            batch_group_size=0,
            min_seq_len=3,
            max_seq_len=500,
            compute_f0=False,
            use_noise_augment=True,
            compute_input_seq_cache=True,
            output_path=self.output_path,
            datasets=[dataset],
        )
        return config

    def ljspeech_speedy_speech(self, audio, dataset):
        """Base speedy speech model for ljpseech dataset."""
        model_args = SpeedySpeechArgs(
            positional_encoding=True,
            hidden_channels=128,
            encoder_type="residual_conv_bn",
            decoder_type="residual_conv_bn",
        )
        config = SpeedySpeechConfig(
            audio=audio,
            run_name="ljpseech-speedy-speech",
            run_description="speedy-speech model for LJSpeech dataset",
            model_args=model_args,
            batch_size=self.batch_size,
            eval_batch_size=self.batch_size // 2,
            r=1,
            loss_masking=True,
            ssim_alpha=1,
            l1_alpha=1,
            huber_alpha=1,
            run_eval=True,
            test_delay_epochs=-1,
            grad_clip=1.0,
            epochs=self.epochs,
            lr=self.learning_rate,
            print_step=25,
            plot_step=100,
            print_eval=False,
            save_step=5000,
            checkpoint=True,
            model_param_stats=False,
            mixed_precision=self.mixed_precision,
            text_cleaner="phoneme_cleaners",
            enable_eos_bos_chars=False,
            num_loader_workers=8,
            batch_group_size=4,
            min_seq_len=2,
            max_seq_len=300,
            compute_f0=False,
            compute_input_seq_cache=True,
            output_path=self.output_path,
            phoneme_cache_path=os.path.join(self.output_path, "phoneme_cache"),
            use_phonemes=True,
            phoneme_language="en-us",
            use_speaker_embedding=False,
            datasets=[dataset],
        )
        return config

    def sc_glow_tts(self, audio, dataset, speaker_file, encoder):
        if encoder == "transformer":
            encoder_type = "rel_pos_transformer"
        elif encoder == "gated":
            encoder_type = "gated_conv"
        elif encoder == "residual_bn":
            encoder_type = "residual_conv_bn"
        elif encoder == "time_depth":
            encoder_type = "time_depth_separable"
        config = GlowTTSConfig(
            audio=audio,
            run_name="multispeaker glow tts",
            run_description="glow tts for multispeaker datasets",
            batch_size=self.batch_size,
            eval_batch_size=self.batch_size // 2,
            mixed_precision=self.mixed_precision,
            run_eval=True,
            test_delay_epochs=-1,
            print_eval=False,
            print_step=25,
            plot_step=100,
            model_param_stats=False,
            save_step=10000,
            num_loader_workers=8,
            num_eval_loader_workers=8,
            use_noise_augment=False,
            output_path=self.output_path,
            use_phonemes=True,
            use_espeak_phonemes=True,
            phoneme_language="en",
            compute_input_seq_cache=False,
            test_sentences_file=None,
            phoneme_cache_path=os.path.join(self.output_path, "phoneme_cache"),
            batch_group_size=0,
            loss_masking=True,
            min_seq_len=2,
            max_seq_len=500,
            compute_f0=False,
            add_blank=True,
            use_speaker_embedding=True,
            use_d_vector_file=True,
            d_vector_dim=256,
            encoder_type=encoder_type,
            use_encoder_prenet=True,
            hidden_channels_dp=256,
            hidden_channels_dec=192,
            hidden_channels_enc=192,
            dropout_p_dp=0.1,
            dropout_p_dec=0.05,
            mean_only=True,
            out_channels=80,
            num_flow_blocks_dec=12,
            inference_noise_scale=0.0,
            kernel_size_dec=5,
            dilation_rate=1,
            num_block_layers=4,
            num_speakers=0,
            num_splits=4,
            num_squeeze=2,
            sigmoid_scale=False,
            data_dep_init_steps=10,
            style_wav_for_test=None,
            length_scale=1.0,
            d_vector_file=speaker_file,
            grad_clip=5.0,
            lr=self.learning_rate,
            r=1,
            datasets=[dataset],
        )
        return config

    def ljspeech_vits_tts(self, audio, dataset):
        config = VitsConfig(
            audio=audio,
            run_name="vits_ljspeech",
            batch_size=self.batch_size,
            eval_batch_size=self.batch_size // 2,
            batch_group_size=0,
            num_loader_workers=4,
            num_eval_loader_workers=4,
            run_eval=True,
            test_delay_epochs=-1,
            epochs=self.epochs,
            text_cleaner="english_cleaners",
            use_phonemes=True,
            phoneme_language="en-us",
            phoneme_cache_path=os.path.join(self.output_path, "phoneme_cache"),
            compute_input_seq_cache=True,
            print_step=25,
            print_eval=True,
            mixed_precision=self.mixed_precision,
            max_seq_len=5000,
            output_path=self.output_path,
            datasets=[dataset],
        )
        return config


# ToDo: test these models and tune config if needed
class VocoderModels:
    def __init__(
        self,
        batch_size,
        mixed_precision,
        generator_learning_rate,
        discriminator_learning_rate,
        epochs,
        output_path=os.path.dirname(os.path.abspath(__file__)),
    ):
        self.batch_size = batch_size
        self.output_path = output_path
        self.mixed_precision = mixed_precision
        self.generator_learning_rate = generator_learning_rate
        self.discriminator_lr = discriminator_learning_rate
        self.epochs = epochs

    @staticmethod
    def loss_func(loss=None):
        if loss == "mse":
            pass  # I was about to impliment a way to just pick a loss func but I wanna think of a better way to do it so ima just leave this here for now.

    def ljspeech_hifi_gan(self, audio, data_path):
        config = HifiganConfig(
            audio=audio,
            run_name="ljspeech-hifigan",
            run_description="hifi gan vocoder model trained on the ljspeech dataset.",
            batch_size=self.batch_size,
            eval_batch_size=self.batch_size // 2,
            num_loader_workers=4,
            num_eval_loader_workers=4,
            run_eval=True,
            test_delay_epochs=-1,
            epochs=self.epochs,
            seq_len=8192,
            pad_short=2000,
            use_noise_augment=True,
            eval_split_size=10,
            print_step=25,
            print_eval=True,
            mixed_precision=self.mixed_precision,
            lr_gen=self.generator_learning_rate,
            lr_disc=self.discriminator_lr,
            use_pqmf=False,
            use_stft_loss=False,
            use_subband_stft_loss=False,
            use_mse_gan_loss=True,
            use_hinge_gan_loss=False,
            use_l1_spec_loss=True,
            stft_loss_weight=0,
            subband_stft_loss_weight=0,
            mse_G_loss_weight=1,
            hinge_G_loss_weight=0,
            feat_match_loss_weight=10,
            l1_spec_loss_weight=45,
            target_loss="avg_G_loss",
            data_path=data_path,
            output_path=self.output_path,
        )
        return config

    def ljspeech_wave_grad(self, audio, data_path):
        config = WavegradConfig(
            audio=audio,
            batch_size=self.batch_size,
            eval_batch_size=self.batch_size // 2,
            num_loader_workers=4,
            num_eval_loader_workers=4,
            run_eval=True,
            test_delay_epochs=-1,
            epochs=self.epochs,
            seq_len=6144,
            pad_short=2000,
            use_noise_augment=True,
            eval_split_size=50,
            print_step=50,
            lr=self.generator_learning_rate,
            print_eval=True,
            mixed_precision=self.mixed_precision,
            data_path=data_path,
            output_path=self.output_path,
        )
        return config

    def ljspeech_multiband_mel_gan(self, audio, data_path):
        config = MultibandMelganConfig(
            audio=audio,
            batch_size=self.batch_size,
            eval_batch_size=self.batch_size // 2,
            num_loader_workers=4,
            num_eval_loader_workers=4,
            run_eval=True,
            test_delay_epochs=-1,
            epochs=self.epochs,
            seq_len=8192,
            pad_short=2000,
            use_noise_augment=False,
            use_stft_loss=True,
            use_feat_match_loss=False,
            feat_match_loss_weight=25,
            subband_stft_loss_weight=0.5,
            eval_split_size=20,
            print_step=25,
            print_eval=True,
            mixed_precision=self.mixed_precision,
            lr_gen=self.generator_learning_rate,
            lr_disc=self.discriminator_lr,
            data_path=data_path,
            output_path=self.output_path,
        )
        return config

    def ljspeech_univnet(self, audio, data_path):
        config = UnivnetConfig(
            audio=audio,
            batch_size=self.batch_size,
            eval_batch_size=self.batch_size // 2,
            num_loader_workers=4,
            num_eval_loader_workers=4,
            run_eval=True,
            test_delay_epochs=-1,
            epochs=self.epochs,
            use_cache=False,
            wd=0.0,
            conv_pad=0,
            use_stft_loss=True,
            use_l1_spec_loss=False,
            # ToDo: make function that lets you pick if you want to use the one from the original paper or mse
            use_mse_gan_loss=True,
            target_loss="loss_0",
            grad_clip=[5.0, 5.0],
            lr_gen=self.generator_learning_rate,
            lr_disc=self.discriminator_lr,
            use_pqmf=False,
            diff_samples_for_G_and_D=False,
            seq_len=8192,
            pad_short=2000,
            use_noise_augment=True,
            eval_split_size=10,
            print_step=25,
            print_eval=False,
            mixed_precision=False,
            data_path=data_path,
            output_path=self.output_path,
        )
        return config

    def ljspeechUnivnet(self, audio, data_path):
        config = UnivnetConfig(
            audio=audio,
            batch_size=self.batch_size,
            eval_batch_size=self.batch_size // 2,
            num_loader_workers=4,
            num_eval_loader_workers=4,
            run_eval=True,
            test_delay_epochs=-1,
            epochs=self.epochs,
            use_cache=False,
            wd=0.0,
            conv_pad=0,
            use_stft_loss=True,
            use_l1_spec_loss=False,
            # ToDo: make function that lets you pick if you want to use the one from the original paper or mse
            use_mse_gan_loss=True,
            target_loss="loss_0",
            grad_clip=[5.0, 5.0],
            lr_gen=self.generator_learning_rate,
            lr_disc=self.discriminator_lr,
            use_pqmf=False,
            diff_samples_for_G_and_D=False,
            seq_len=8192,
            pad_short=2000,
            use_noise_augment=True,
            eval_split_size=10,
            print_step=25,
            print_eval=False,
            mixed_precision=False,
            data_path=data_path,
            output_path=self.output_path,
        )
        return config
