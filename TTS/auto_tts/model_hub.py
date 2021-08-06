import os

from TTS.tts.configs.glow_tts_config import GlowTTSConfig
from TTS.tts.configs.speedy_speech_config import SpeedySpeechArgs, SpeedySpeechConfig
from TTS.tts.configs.tacotron2_config import Tacotron2Config


class Models:
    """
    This is a class that holds all the model configs.
    If you want to add a new recipe this is where you would add your model config
    made this so the complete recipes file doesnt look so cluttered.

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

    def single_speaker_tacotron2_base(self, audio, dataset):
        config = Tacotron2Config(
            run_name="single_speaker_taoctron2",
            audio=audio,
            batch_size=self.batch_size,
            eval_batch_size=int(self.batch_size / 2),
            r=2,
            grad_clip=1,
            lr=self.learning_rate,
            memory_size=-1,
            prenet_type="original",
            prenet_dropout=True,
            attention_type="original",
            attention_heads=5,
            attention_norm="sigmoid",
            windowing=False,
            use_forward_attn=True,
            forward_attn_mask=False,
            transition_agent=False,
            location_attn=True,
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

    def single_speaker_tacotron2_DDC(self, audio, dataset):
        config = Tacotron2Config(
            audio=audio,
            run_name="ljspeech-ddc",
            run_description="tacotron2 with double decoder consistency.",
            batch_size=self.batch_size,
            eval_batch_size=self.batch_size // 2,
            mixed_precision=False,
            loss_masking=True,
            decoder_loss_alpha=0.25,
            postnet_loss_alpha=0.25,
            postnet_diff_spec_alpha=0.25,
            decoder_diff_spec_alpha=0.25,
            decoder_ssim_alpha=0.25,
            postnet_ssim_alpha=0.25,
            ga_alpha=5.0,
            stopnet_pos_weight=15.0,
            run_eval=True,
            test_delay_epochs=10,
            max_decoder_steps=1000,
            grad_clip=0.05,
            epochs=self.epochs,
            lr=self.learning_rate,
            memory_size=-1,
            prenet_type="original",
            prenet_dropout=True,
            attention_type="original",
            location_attn=True,
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
            tb_plot_step=100,
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
            phoneme_cache_path=os.path.join(self.output_path, "phoneme_cache"),
            output_path=self.output_path,
            use_phonemes=False,
            phoneme_language="en-us",
            datasets=[dataset],
        )
        return config

    def single_speaker_tacotron2_DCA(self, audio, dataset):
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
            decoder_loss_alpha=0.25,
            postnet_loss_alpha=0.25,
            postnet_diff_spec_alpha=0.25,
            decoder_diff_spec_alpha=0.25,
            decoder_ssim_alpha=0.25,
            postnet_ssim_alpha=0.25,
            ga_alpha=5.0,
            stopnet_pos_weight=15.0,
            run_eval=True,
            test_delay_epochs=10,
            max_decoder_steps=1000,
            grad_clip=0.05,
            epochs=self.epochs,
            lr=self.learning_rate,
            memory_size=-1,
            prenet_type="original",
            prenet_dropout=True,
            attention_type="dynamic_convolution",
            location_attn=True,
            attention_norm="sigmoid",
            r=2,
            stopnet=True,
            separate_stopnet=True,
            print_step=25,
            tb_plot_step=100,
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

    def ljspeech_glow_tts(self, audio, dataset, encoder_type="rel_pos_transformer"):
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
            tb_plot_step=100,
            print_eval=False,
            save_step=5000,
            checkpoint=True,
            tb_model_param_stats=False,
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
