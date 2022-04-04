from dataclasses import dataclass, field
from itertools import chain
from typing import Dict, List, Tuple, Union

import torch
from coqpit import Coqpit
from torch import nn
from torch.cuda.amp.autocast_mode import autocast
from trainer.trainer_utils import get_optimizer, get_scheduler

from TTS.tts.layers.losses import ForwardTTSE2ELoss, VitsDiscriminatorLoss
from TTS.tts.layers.vits.discriminator import VitsDiscriminator
from TTS.tts.models.base_tts import BaseTTSE2E
from TTS.tts.models.forward_tts import ForwardTTS, ForwardTTSArgs
from TTS.tts.models.vits import wav_to_mel
from TTS.tts.utils.helpers import rand_segments, segment
from TTS.tts.utils.speakers import SpeakerManager
from TTS.tts.utils.synthesis import synthesis
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.tts.utils.visual import plot_alignment
from TTS.vocoder.models.hifigan_generator import HifiganGenerator
from TTS.vocoder.utils.generic_utils import plot_results


@dataclass
class ForwardTTSE2EArgs(ForwardTTSArgs):
    # vocoder_config: BaseGANVocoderConfig = None
    num_chars: int = 100
    encoder_out_channels: int = 80
    spec_segment_size: int = 32
    # duration predictor
    detach_duration_predictor: bool = True
    # discriminator
    init_discriminator: bool = True
    use_spectral_norm_discriminator: bool = False
    # model parameters
    detach_vocoder_input: bool = False
    hidden_channels: int = 192
    encoder_type: str = "fftransformer"
    encoder_params: dict = field(
        default_factory=lambda: {"hidden_channels_ffn": 768, "num_heads": 2, "num_layers": 6, "dropout_p": 0.1}
    )
    decoder_type: str = "fftransformer"
    decoder_params: dict = field(
        default_factory=lambda: {"hidden_channels_ffn": 768, "num_heads": 2, "num_layers": 6, "dropout_p": 0.1}
    )
    # generator
    resblock_type_decoder: str = "1"
    resblock_kernel_sizes_decoder: List[int] = field(default_factory=lambda: [3, 7, 11])
    resblock_dilation_sizes_decoder: List[List[int]] = field(default_factory=lambda: [[1, 3, 5], [1, 3, 5], [1, 3, 5]])
    upsample_rates_decoder: List[int] = field(default_factory=lambda: [8, 8, 2, 2])
    upsample_initial_channel_decoder: int = 512
    upsample_kernel_sizes_decoder: List[int] = field(default_factory=lambda: [16, 16, 4, 4])
    # multi-speaker params
    use_speaker_embedding: bool = False
    num_speakers: int = 0
    speakers_file: str = None
    d_vector_file: str = None
    speaker_embedding_channels: int = 384
    use_d_vector_file: bool = False
    d_vector_dim: int = 0


class ForwardTTSE2E(BaseTTSE2E):
    """
    Model training::
        text --> ForwardTTS() --> spec_hat --> rand_seg_select()--> GANVocoder() --> waveform_seg
        spec --------^

    Examples:
        >>> from TTS.tts.models.forward_tts_e2e import ForwardTTSE2E, ForwardTTSE2EConfig
        >>> config = ForwardTTSE2EConfig()
        >>> model = ForwardTTSE2E(config)
    """

    # pylint: disable=dangerous-default-value
    def __init__(
        self,
        config: Coqpit,
        ap: "AudioProcessor" = None,
        tokenizer: "TTSTokenizer" = None,
        speaker_manager: SpeakerManager = None,
    ):
        super().__init__(config, ap, tokenizer, speaker_manager)
        self._set_model_args(config)

        self.init_multispeaker(config)

        self.encoder_model = ForwardTTS(config=self.args, ap=ap, tokenizer=tokenizer, speaker_manager=speaker_manager)
        # self.vocoder_model = GAN(config=self.args.vocoder_config, ap=ap)
        self.waveform_decoder = HifiganGenerator(
            self.args.out_channels,
            1,
            self.args.resblock_type_decoder,
            self.args.resblock_dilation_sizes_decoder,
            self.args.resblock_kernel_sizes_decoder,
            self.args.upsample_kernel_sizes_decoder,
            self.args.upsample_initial_channel_decoder,
            self.args.upsample_rates_decoder,
            inference_padding=0,
            cond_channels=self.embedded_speaker_dim,
            conv_pre_weight_norm=False,
            conv_post_weight_norm=False,
            conv_post_bias=False,
        )

        # use Vits Discriminator for limiting VRAM use
        if self.args.init_discriminator:
            self.disc = VitsDiscriminator(use_spectral_norm=self.args.use_spectral_norm_discriminator)

    def init_multispeaker(self, config: Coqpit):
        """Init for multi-speaker training.

        Args:
            config (Coqpit): Model configuration.
        """
        self.embedded_speaker_dim = 0
        self.num_speakers = self.args.num_speakers
        self.audio_transform = None

        if self.speaker_manager:
            self.num_speakers = self.speaker_manager.num_speakers

        if self.args.use_speaker_embedding:
            self._init_speaker_embedding()

        if self.args.use_d_vector_file:
            self._init_d_vector()

    def _init_speaker_embedding(self):
        # pylint: disable=attribute-defined-outside-init
        if self.num_speakers > 0:
            print(" > initialization of speaker-embedding layers.")
            self.embedded_speaker_dim = self.args.speaker_embedding_channels
            self.emb_g = nn.Embedding(self.num_speakers, self.embedded_speaker_dim)

    def _init_d_vector(self):
        # pylint: disable=attribute-defined-outside-init
        if hasattr(self, "emb_g"):
            raise ValueError("[!] Speaker embedding layer already initialized before d_vector settings.")
        self.embedded_speaker_dim = self.args.d_vector_dim

    def get_aux_input(self, *args, **kwargs) -> Dict:
        return self.encoder_model.get_aux_input(*args, **kwargs)

    def forward(
        self,
        x: torch.LongTensor,
        x_lengths: torch.LongTensor,
        spec_lengths: torch.LongTensor,
        spec: torch.FloatTensor,
        waveform: torch.FloatTensor,
        dr: torch.IntTensor = None,
        pitch: torch.FloatTensor = None,
        aux_input: Dict = {"d_vectors": None, "speaker_ids": None},  # pylint: disable=unused-argument
    ) -> Dict:
        """Model's forward pass.

        Args:
            x (torch.LongTensor): Input character sequences.
            x_lengths (torch.LongTensor): Input sequence lengths.
            spec_lengths (torch.LongTensor): Spectrogram sequnce lengths. Defaults to None.
            spec (torch.FloatTensor): Spectrogram frames. Only used when the alignment network is on. Defaults to None.
            waveform (torch.FloatTensor): Waveform. Defaults to None.
            dr (torch.IntTensor): Character durations over the spectrogram frames. Only used when the alignment network is off. Defaults to None.
            pitch (torch.FloatTensor): Pitch values for each spectrogram frame. Only used when the pitch predictor is on. Defaults to None.
            aux_input (Dict): Auxiliary model inputs for multi-speaker training. Defaults to `{"d_vectors": 0, "speaker_ids": None}`.

        Shapes:
            - x: :math:`[B, T_max]`
            - x_lengths: :math:`[B]`
            - spec_lengths: :math:`[B]`
            - spec: :math:`[B, T_max2]`
            - waveform: :math:`[B, C, T_max2]`
            - dr: :math:`[B, T_max]`
            - g: :math:`[B, C]`
            - pitch: :math:`[B, 1, T]`
        """
        encoder_outputs = self.encoder_model(
            x=x, x_lengths=x_lengths, y_lengths=spec_lengths, y=spec, dr=dr, pitch=pitch, aux_input=aux_input
        )
        spec_encoder_output = encoder_outputs["model_outputs"]
        spec_encoder_output_slices, slice_ids = rand_segments(
            x=spec_encoder_output.transpose(1, 2),
            x_lengths=spec_lengths,
            segment_size=self.args.spec_segment_size,
            let_short_samples=True,
            pad_short=True,
        )
        vocoder_output = self.waveform_decoder(
            x=spec_encoder_output_slices.detach() if self.args.detach_vocoder_input else spec_encoder_output_slices,
            g=encoder_outputs["g"],
        )
        wav_seg = segment(
            waveform,
            slice_ids * self.config.audio.hop_length,
            self.args.spec_segment_size * self.config.audio.hop_length,
            pad_short=True,
        )
        model_outputs = {**encoder_outputs}
        model_outputs["encoder_outputs"] = encoder_outputs["model_outputs"]
        model_outputs["model_outputs"] = vocoder_output
        model_outputs["waveform_seg"] = wav_seg
        model_outputs["slice_ids"] = slice_ids
        return model_outputs

    @torch.no_grad()
    def inference(self, x, aux_input={"d_vectors": None, "speaker_ids": None}):  # pylint: disable=unused-argument
        encoder_outputs = self.encoder_model.inference(x=x, aux_input=aux_input)
        # vocoder_output = self.vocoder_model.model_g(x=encoder_outputs["model_outputs"].transpose(1, 2))
        vocoder_output = self.waveform_decoder(
            x=encoder_outputs["model_outputs"].transpose(1, 2), g=encoder_outputs["g"]
        )
        model_outputs = {**encoder_outputs}
        model_outputs["encoder_outputs"] = encoder_outputs["model_outputs"]
        model_outputs["model_outputs"] = vocoder_output
        return model_outputs

    @staticmethod
    def init_from_config(config: "ForwardTTSConfig", samples: Union[List[List], List[Dict]] = None, verbose=False):
        """Initiate model from config

        Args:
            config (ForwardTTSE2EConfig): Model config.
            samples (Union[List[List], List[Dict]]): Training samples to parse speaker ids for training.
                Defaults to None.
        """
        from TTS.utils.audio import AudioProcessor

        ap = AudioProcessor.init_from_config(config, verbose=verbose)
        tokenizer, new_config = TTSTokenizer.init_from_config(config)
        speaker_manager = SpeakerManager.init_from_config(config, samples)
        # language_manager = LanguageManager.init_from_config(config)
        return ForwardTTSE2E(config=new_config, ap=ap, tokenizer=tokenizer, speaker_manager=speaker_manager)

    def load_checkpoint(
        self, config, checkpoint_path, eval=False
    ):  # pylint: disable=unused-argument, redefined-builtin
        state = torch.load(checkpoint_path, map_location=torch.device("cpu"))
        self.load_state_dict(state["model"])
        if eval:
            self.eval()
            assert not self.training

    def train_step(self, batch: dict, criterion: nn.Module, optimizer_idx: int):
        if optimizer_idx == 0:
            tokens = batch["text_input"]
            token_lenghts = batch["text_lengths"]
            spec = batch["mel_input"]
            spec_lens = batch["mel_lengths"]
            waveform = batch["waveform"].transpose(1, 2)  # [B, T, C] -> [B, C, T]
            pitch = batch["pitch"]
            d_vectors = batch["d_vectors"]
            speaker_ids = batch["speaker_ids"]
            language_ids = batch["language_ids"]

            # generator pass
            outputs = self.forward(
                x=tokens,
                x_lengths=token_lenghts,
                spec_lengths=spec_lens,
                spec=spec,
                waveform=waveform,
                pitch=pitch,
                aux_input={"d_vectors": d_vectors, "speaker_ids": speaker_ids, "language_ids": language_ids},
            )

            # cache tensors for the generator pass
            self.model_outputs_cache = outputs  # pylint: disable=attribute-defined-outside-init

            # compute scores and features
            scores_d_fake, _, scores_d_real, _ = self.disc(outputs["model_outputs"].detach(), outputs["waveform_seg"])

            # compute loss
            with autocast(enabled=False):  # use float32 for the criterion
                loss_dict = criterion[optimizer_idx](
                    scores_disc_fake=scores_d_fake,
                    scores_disc_real=scores_d_real,
                )
            return outputs, loss_dict

        if optimizer_idx == 1:
            mel = batch["mel_input"].transpose(1, 2)

            # compute melspec segment
            with autocast(enabled=False):
                mel_slice = segment(
                    mel.float(), self.model_outputs_cache["slice_ids"], self.args.spec_segment_size, pad_short=True
                )

                mel_slice_hat = wav_to_mel(
                    y=self.model_outputs_cache["model_outputs"].float(),
                    n_fft=self.config.audio.fft_size,
                    sample_rate=self.config.audio.sample_rate,
                    num_mels=self.config.audio.num_mels,
                    hop_length=self.config.audio.hop_length,
                    win_length=self.config.audio.win_length,
                    fmin=self.config.audio.mel_fmin,
                    fmax=self.config.audio.mel_fmax,
                    center=False,
                )

            # compute discriminator scores and features
            scores_d_fake, feats_d_fake, _, feats_d_real = self.disc(
                self.model_outputs_cache["model_outputs"], self.model_outputs_cache["waveform_seg"]
            )

            # compute losses
            with autocast(enabled=False):  # use float32 for the criterion
                loss_dict = criterion[optimizer_idx](
                    decoder_output=self.model_outputs_cache["encoder_outputs"],
                    decoder_target=batch["mel_input"],
                    decoder_output_lens=batch["mel_lengths"],
                    dur_output=self.model_outputs_cache["durations_log"],
                    dur_target=self.model_outputs_cache["aligner_durations"],
                    pitch_output=self.model_outputs_cache["pitch_avg"] if self.args.use_pitch else None,
                    pitch_target=self.model_outputs_cache["pitch_avg_gt"] if self.args.use_pitch else None,
                    input_lens=batch["text_lengths"],
                    aligner_logprob=self.model_outputs_cache["aligner_logprob"],
                    aligner_hard=self.model_outputs_cache["aligner_mas"],
                    aligner_soft=self.model_outputs_cache["aligner_soft"],
                    binary_loss_weight=self.encoder_model.binary_loss_weight,
                    feats_fake=feats_d_fake,
                    feats_real=feats_d_real,
                    scores_fake=scores_d_fake,
                    spec_slice=mel_slice,
                    spec_slice_hat=mel_slice_hat,
                )

                # compute duration error for logging
                durations_pred = self.model_outputs_cache["durations"]
                durations_target = self.model_outputs_cache["aligner_durations"]
                duration_error = torch.abs(durations_target - durations_pred).sum() / batch["text_lengths"].sum()
                loss_dict["duration_error"] = duration_error

            return self.model_outputs_cache, loss_dict

        raise ValueError(" [!] Unexpected `optimizer_idx`.")

    def eval_step(self, batch: dict, criterion: nn.Module, optimizer_idx: int):
        return self.train_step(batch, criterion, optimizer_idx)

    @staticmethod
    def __copy_for_logging(outputs):
        """Change keys and copy values for logging."""
        encoder_outputs = outputs[1].copy()
        encoder_outputs["model_outputs"] = encoder_outputs["encoder_outputs"]
        vocoder_outputs = outputs.copy()
        vocoder_outputs[1]["model_outputs"] = outputs[1]["model_outputs"]
        return encoder_outputs, vocoder_outputs

    def _log(self, ap, batch, outputs, name_prefix="train"):
        encoder_outputs, vocoder_outputs = self.__copy_for_logging(outputs)
        y_hat = vocoder_outputs[1]["model_outputs"]
        y = vocoder_outputs[1]["waveform_seg"]
        # encoder outputs
        encoder_figures, encoder_audios = self.encoder_model.create_logs(
            batch=batch, outputs=encoder_outputs, ap=self.ap
        )
        # vocoder outputs
        vocoder_figures = plot_results(y_hat, y, ap, name_prefix)
        sample_voice = y_hat[0].squeeze(0).detach().cpu().numpy()
        audios = {f"{name_prefix}/real_audio": sample_voice}
        audios[f"{name_prefix}/encoder_audio"] = encoder_audios["audio"]
        figures = {**encoder_figures, **vocoder_figures}
        return figures, audios

    def train_log(
        self, batch: dict, outputs: dict, logger: "Logger", assets: dict, steps: int
    ):  # pylint: disable=no-self-use, unused-argument
        """Create visualizations and waveform examples.

        For example, here you can plot spectrograms and generate sample sample waveforms from these spectrograms to
        be projected onto Tensorboard.

        Args:
            ap (AudioProcessor): audio processor used at training.
            batch (Dict): Model inputs used at the previous training step.
            outputs (Dict): Model outputs generated at the previous training step.

        Returns:
            Tuple[Dict, np.ndarray]: training plots and output waveform.
        """
        figures, audios = self._log(ap=self.ap, batch=batch, outputs=outputs, name_prefix="vocoder/")
        logger.train_figures(steps, figures)
        logger.train_audios(steps, audios, self.ap.sample_rate)

    def eval_log(self, batch: dict, outputs: dict, logger: "Logger", assets: dict, steps: int) -> None:
        figures, audios = self._log(ap=self.ap, batch=batch, outputs=outputs, name_prefix="vocoder/")
        logger.eval_figures(steps, figures)
        logger.eval_audios(steps, audios, self.ap.sample_rate)

    def get_aux_input_from_test_sentences(self, sentence_info):
        if hasattr(self.config, "model_args"):
            config = self.config.model_args
        else:
            config = self.config

        # extract speaker and language info
        text, speaker_name, style_wav, language_name = None, None, None, None  # pylint: disable=unused-variable

        if isinstance(sentence_info, list):
            if len(sentence_info) == 1:
                text = sentence_info[0]
            elif len(sentence_info) == 2:
                text, speaker_name = sentence_info
            elif len(sentence_info) == 3:
                text, speaker_name, style_wav = sentence_info
            elif len(sentence_info) == 4:
                text, speaker_name, style_wav, language_name = sentence_info
        else:
            text = sentence_info

        # get speaker  id/d_vector
        speaker_id, d_vector, language_id = None, None, None  # pylint: disable=unused-variable
        if hasattr(self, "speaker_manager"):
            if config.use_d_vector_file:
                if speaker_name is None:
                    d_vector = self.speaker_manager.get_random_d_vector()
                else:
                    d_vector = self.speaker_manager.get_mean_d_vector(speaker_name, num_samples=None, randomize=False)
            elif config.use_speaker_embedding:
                if speaker_name is None:
                    speaker_id = self.speaker_manager.get_random_speaker_id()
                else:
                    speaker_id = self.speaker_manager.speaker_ids[speaker_name]

        # get language id
        # if hasattr(self, "language_manager") and config.use_language_embedding and language_name is not None:
        # language_id = self.language_manager.language_id_mapping[language_name]

        return {
            "text": text,
            "speaker_id": speaker_id,
            "style_wav": style_wav,
            "d_vector": d_vector,
            "language_id": None,
            "language_name": None,
        }

    @torch.no_grad()
    def test_run(self, assets) -> Tuple[Dict, Dict]:
        """Generic test run for `tts` models used by `Trainer`.

        You can override this for a different behaviour.

        Returns:
            Tuple[Dict, Dict]: Test figures and audios to be projected to Tensorboard.
        """
        print(" | > Synthesizing test sentences.")
        test_audios = {}
        test_figures = {}
        test_sentences = self.config.test_sentences
        for idx, s_info in enumerate(test_sentences):
            aux_inputs = self.get_aux_input_from_test_sentences(s_info)
            wav, alignment, _, _ = synthesis(
                self,
                aux_inputs["text"],
                self.config,
                "cuda" in str(next(self.parameters()).device),
                speaker_id=aux_inputs["speaker_id"],
                d_vector=aux_inputs["d_vector"],
                style_wav=aux_inputs["style_wav"],
                language_id=aux_inputs["language_id"],
                use_griffin_lim=True,
                do_trim_silence=False,
            ).values()
            test_audios["{}-audio".format(idx)] = wav
            test_figures["{}-alignment".format(idx)] = plot_alignment(alignment, output_fig=False)
        return {"figures": test_figures, "audios": test_audios}

    def test_log(
        self, outputs: dict, logger: "Logger", assets: dict, steps: int  # pylint: disable=unused-argument
    ) -> None:
        logger.test_audios(steps, outputs["audios"], self.ap.sample_rate)
        logger.test_figures(steps, outputs["figures"])

    def get_criterion(self):
        return [VitsDiscriminatorLoss(self.config), ForwardTTSE2ELoss(self.config)]

    def get_optimizer(self) -> List:
        """Initiate and return the GAN optimizers based on the config parameters.
        It returnes 2 optimizers in a list. First one is for the generator and the second one is for the discriminator.
        Returns:
            List: optimizers.
        """
        optimizer0 = get_optimizer(self.config.optimizer, self.config.optimizer_params, self.config.lr_disc, self.disc)
        gen_parameters = chain(params for k, params in self.named_parameters() if not k.startswith("disc."))
        optimizer1 = get_optimizer(
            self.config.optimizer, self.config.optimizer_params, self.config.lr_gen, parameters=gen_parameters
        )
        return [optimizer0, optimizer1]

    def get_lr(self) -> List:
        """Set the initial learning rates for each optimizer.

        Returns:
            List: learning rates for each optimizer.
        """
        return [self.config.lr_disc, self.config.lr_gen]

    def get_scheduler(self, optimizer) -> List:
        """Set the schedulers for each optimizer.

        Args:
            optimizer (List[`torch.optim.Optimizer`]): List of optimizers.

        Returns:
            List: Schedulers, one for each optimizer.
        """
        scheduler_D = get_scheduler(self.config.lr_scheduler_gen, self.config.lr_scheduler_gen_params, optimizer[0])
        scheduler_G = get_scheduler(self.config.lr_scheduler_disc, self.config.lr_scheduler_disc_params, optimizer[1])
        return [scheduler_D, scheduler_G]

    def on_train_step_start(self, trainer):
        """Schedule binary loss weight."""
        self.encoder_model.config.binary_loss_warmup_epochs = self.config.binary_loss_warmup_epochs
        self.encoder_model.on_train_step_start(trainer)
