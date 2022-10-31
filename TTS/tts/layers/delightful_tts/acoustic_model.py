from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from coqpit import Coqpit
from torch import nn
from TTS.tts.layers.delightful_tts import energy_adapter, phoneme_prosody_predictor
from TTS.tts.layers.delightful_tts import variance_predictor
from TTS.tts.layers.delightful_tts.conformer import Conformer
from TTS.tts.layers.delightful_tts.networks import EmbeddingPadded, positional_encoding
from TTS.tts.layers.delightful_tts.encoders import PhonemeLevelProsodyEncoder, UtteranceLevelProsodyEncoder, get_mask_from_lengths
from TTS.tts.layers.delightful_tts.pitch_adapter import PitchAdaptor
from TTS.tts.layers.delightful_tts.variance_predictor import VariancePredictor, VariancePredictorLSTM
from TTS.tts.layers.delightful_tts.phoneme_prosody_predictor import PhonemeProsodyPredictor
from TTS.tts.layers.delightful_tts.energy_adapter import EnergyAdaptor

from TTS.tts.layers.generic.aligner import AlignmentNetwork
from TTS.tts.utils.emotions import EmotionManager
from TTS.tts.utils.helpers import generate_path, maximum_path, sequence_mask
from TTS.tts.utils.speakers import SpeakerManager
from TTS.tts.utils.text.tokenizer import TTSTokenizer


class AcousticModel(torch.nn.Module):
    def __init__(
        self,
        args: "DelightfulTtsArgs",
        tokenizer: "TTSTokenizer" = None,
        speaker_manager: SpeakerManager = None,
        emotion_manager: EmotionManager = None,
    ):
        super().__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.speaker_manager = speaker_manager
        self.emotion_manager = emotion_manager

        self.init_multispeaker(args)
        self.init_emotion(emotion_manager)
        self.set_embedding_dims()

        self.length_scale = (
            float(self.args.length_scale) if isinstance(self.args.length_scale, int) else self.args.length_scale
        )

        self.emb_dim = args.acoustic_config.encoder.n_hidden
        self.encoder = Conformer(
            dim=args.acoustic_config.encoder.n_hidden,
            n_layers=args.acoustic_config.encoder.n_layers,
            n_heads=args.acoustic_config.encoder.n_heads,
            speaker_embedding_dim=self.embedded_speaker_dim,
            emotion_embedding_dim=self.embedded_emotion_dim,
            p_dropout=args.acoustic_config.encoder.p_dropout,
            kernel_size_conv_mod=args.acoustic_config.encoder.kernel_size_conv_mod,
        )
        self.pitch_adaptor = PitchAdaptor(args.acoustic_config)
        self.energy_adaptor = EnergyAdaptor(args.acoustic_config)

        self.aligner = AlignmentNetwork(
            in_query_channels=self.args.acoustic_config.out_channels,
            in_key_channels=args.acoustic_config.encoder.n_hidden,
        )

        # self.duration_predictor = VariancePredictorLSTM(
        #     in_dim=args.acoustic_config.encoder.n_hidden,
        #     out_dim=1,
        #     reduction_factor=4,
        # )

        self.duration_predictor = VariancePredictor(
            channels_in=args.acoustic_config.encoder.n_hidden,
            channels=args.acoustic_config.variance_adaptor.n_hidden,
            channels_out=1,
            kernel_size=args.acoustic_config.variance_adaptor.kernel_size,
            p_dropout=args.acoustic_config.variance_adaptor.p_dropout,
            lrelu_slope=args.acoustic_config.variance_adaptor.lrelu_slope,

        )

        # self.range_predictor = RangePredictor(
        #     in_channels=args.acoustic_config.encoder.n_hidden,
        #     hidden_channels=args.acoustic_config.encoder.n_hidden // 4,
        # )
        # self.gaussian_upsampling = GaussianUpsampling()

        self.utterance_prosody_encoder = UtteranceLevelProsodyEncoder(args.acoustic_config)
        self.utterance_prosody_predictor = PhonemeProsodyPredictor(args=args.acoustic_config, phoneme_level=False)
        self.phoneme_prosody_encoder = PhonemeLevelProsodyEncoder(args.acoustic_config)
        self.phoneme_prosody_predictor = PhonemeProsodyPredictor(args=args.acoustic_config, phoneme_level=True)
        self.u_bottle_out = nn.Linear(
            args.acoustic_config.reference_encoder.bottleneck_size_u,
            args.acoustic_config.encoder.n_hidden,
        )

        # TODO: change this with IntanceNorm as in StyleTTS
        self.u_norm = nn.LayerNorm(
            args.acoustic_config.reference_encoder.bottleneck_size_u,
            elementwise_affine=False,
        )
        self.p_bottle_out = nn.Linear(
            args.acoustic_config.reference_encoder.bottleneck_size_p,
            args.acoustic_config.encoder.n_hidden,
        )
        # TODO: change this with IntanceNorm as in StyleTTS
        self.p_norm = nn.LayerNorm(
            args.acoustic_config.reference_encoder.bottleneck_size_p,
            elementwise_affine=False,
        )

        self.decoder = Conformer(
            dim=args.acoustic_config.decoder.n_hidden,
            n_layers=args.acoustic_config.decoder.n_layers,
            n_heads=args.acoustic_config.decoder.n_heads,
            speaker_embedding_dim=self.embedded_speaker_dim,
            emotion_embedding_dim=self.embedded_emotion_dim,
            p_dropout=args.acoustic_config.decoder.p_dropout,
            kernel_size_conv_mod=args.acoustic_config.decoder.kernel_size_conv_mod,
        )

        padding_idx = self.tokenizer.characters.pad_id
        self.src_word_emb = EmbeddingPadded(
            self.args.num_chars, args.acoustic_config.encoder.n_hidden, padding_idx=padding_idx
        )
        self.to_mel = nn.Linear(
            args.acoustic_config.decoder.n_hidden,
            args.acoustic_config.num_mels,
        )

        self.energy_scaler = torch.nn.BatchNorm1d(1, affine=False, track_running_stats=True, momentum=None)
        self.energy_scaler.requires_grad_(False)

    def init_multispeaker(self, args: Coqpit):
        """Init for multi-speaker training."""
        self.embedded_speaker_dim = 0
        self.num_speakers = self.args.num_speakers
        self.audio_transform = None

        if self.speaker_manager:
            self.num_speakers = self.speaker_manager.num_speakers

        if self.args.use_speaker_embedding:
            self._init_speaker_embedding()

        if self.args.use_d_vector_file:
            self._init_d_vector()

    def init_emotion(self, emotion_manager: "EmotionManager"):
        # pylint: disable=attribute-defined-outside-init
        """Initialize emotion modules of a model. A model can be trained either with a emotion embedding layer
        or with external `embeddings` computed from a emotion encoder model.

        You must provide a `emotion_manager` at initialization to set up the emotion modules.

        Args:
            emotion_manager (Coqpit): Emotion Manager.
        """
        self.embedded_emotion_dim = 0
        self.emotion_manager = emotion_manager
        self.num_emotions = 0

        if self.emotion_manager:
            self.num_emotions = self.emotion_manager.num_emotions

        if self.args.use_emotion_embedding:
            if self.num_emotions > 0:
                print(" > initialization of emotion-embedding layers.")
                self.emb_emotion = nn.Embedding(self.num_emotions, self.args.emotion_embedding_dim)
                self.embedded_emotion_dim += self.args.emotion_embedding_dim

        if self.args.use_emotion_vector_file:
            self.embedded_emotion_dim += self.args.emotion_vector_dim

    def set_embedding_dims(self):
        if self.embedded_emotion_dim > 0 and self.embedded_speaker_dim > 0:
            self.embedding_dims = [self.embedded_speaker_dim, self.embedded_emotion_dim]
        elif self.embedded_emotion_dim > 0:
            self.embedding_dims = self.embedded_emotion_dim
        elif self.embedded_speaker_dim > 0:
            self.embedding_dims = self.embedded_speaker_dim
        else:
            self.embedding_dims = 0

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

    @staticmethod
    def generate_attn(dr, x_mask, y_mask=None):
        """Generate an attention mask from the linear scale durations.

        Args:
            dr (Tensor): Linear scale durations.
            x_mask (Tensor): Mask for the input (character) sequence.
            y_mask (Tensor): Mask for the output (spectrogram) sequence. Compute it from the predicted durations
                if None. Defaults to None.

        Shapes
           - dr: :math:`(B, T_{en})`
           - x_mask: :math:`(B, T_{en})`
           - y_mask: :math:`(B, T_{de})`
        """
        # compute decode mask from the durations
        if y_mask is None:
            y_lengths = dr.sum(1).long()
            y_lengths[y_lengths < 1] = 1
            y_mask = torch.unsqueeze(sequence_mask(y_lengths, None), 1).to(dr.dtype)
        attn_mask = torch.unsqueeze(x_mask, -1) * torch.unsqueeze(y_mask, 2)
        attn = generate_path(dr, attn_mask.squeeze(1)).to(dr.dtype)
        return attn

    def _expand_encoder_with_gaussian_upsampling(
        self,
        o_en: torch.FloatTensor,
        o_ranges: torch.FloatTensor,
        y_lengths: torch.IntTensor,
        dr: torch.IntTensor,
        x_mask: torch.IntTensor,
        x_lengths: torch.LongTensor,
    ):
        """
        Shapes:
            - o_en: :math:`(B, C, T_src)`
            - o_ranges: :math:`(B, T_src)`
            - y_lengths: :math:`(B)`
            - dr: :math:`(B, T_src)`
            - x_mask: :math:`(B, 1, T_src)`
        """
        y_mask = torch.unsqueeze(sequence_mask(y_lengths, None), 1).to(o_en.dtype)
        # expand o_en with durations
        o_en_ex = self.gaussian_upsampling(x=o_en, durations=dr, vars=o_ranges, x_lengths=x_lengths)  # [B, T_de', C_en]
        attn = self.generate_attn(dr, x_mask, y_mask)  # [B, T_en, T_de']
        # positional encoding
        if hasattr(self, "pos_encoder"):
            o_en_ex = self.pos_encoder(o_en_ex, y_mask)
        return y_mask, o_en_ex, attn.transpose(1, 2)

    def _expand_encoder_with_durations(
        self,
        o_en: torch.FloatTensor,
        dr: torch.IntTensor,
        x_mask: torch.IntTensor,
        y_lengths: torch.IntTensor,
    ):
        y_mask = torch.unsqueeze(sequence_mask(y_lengths, None), 1).to(o_en.dtype)
        attn = self.generate_attn(dr, x_mask, y_mask)
        o_en_ex = torch.einsum("kmn, kjm -> kjn", [attn.float(), o_en])
        return y_mask, o_en_ex, attn.transpose(1, 2)

    def _forward_aligner(
        self,
        x: torch.FloatTensor,
        y: torch.FloatTensor,
        x_mask: torch.IntTensor,
        y_mask: torch.IntTensor,
        attn_priors: torch.FloatTensor,
    ) -> Tuple[torch.IntTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Aligner forward pass.

        1. Compute a mask to apply to the attention map.
        2. Run the alignment network.
        3. Apply MAS to compute the hard alignment map.
        4. Compute the durations from the hard alignment map.

        Args:
            x (torch.FloatTensor): Input sequence.
            y (torch.FloatTensor): Output sequence.
            x_mask (torch.IntTensor): Input sequence mask.
            y_mask (torch.IntTensor): Output sequence mask.
            attn_priors (torch.FloatTensor): Prior for the aligner network map.

        Returns:
            Tuple[torch.IntTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
                Durations from the hard alignment map, soft alignment potentials, log scale alignment potentials,
                hard alignment map.

        Shapes:
            - x: :math:`[B, T_en, C_en]`
            - y: :math:`[B, T_de, C_de]`
            - x_mask: :math:`[B, 1, T_en]`
            - y_mask: :math:`[B, 1, T_de]`
            - attn_priors: :math:`[B, T_de, T_en]`

            - aligner_durations: :math:`[B, T_en]`
            - aligner_soft: :math:`[B, T_de, T_en]`
            - aligner_logprob: :math:`[B, 1, T_de, T_en]`
            - aligner_mas: :math:`[B, T_de, T_en]`
        """
        attn_mask = torch.unsqueeze(x_mask, -1) * torch.unsqueeze(y_mask, 2)  # [B, 1, T_en, T_de]
        aligner_soft, aligner_logprob = self.aligner(y.transpose(1, 2), x.transpose(1, 2), x_mask, attn_priors)
        aligner_mas = maximum_path(
            aligner_soft.squeeze(1).transpose(1, 2).contiguous(), attn_mask.squeeze(1).contiguous()
        )
        aligner_durations = torch.sum(aligner_mas, -1).int()
        aligner_soft = aligner_soft.squeeze(1)  # [B, T_max2, T_max]
        aligner_mas = aligner_mas.transpose(1, 2)  # [B, T_max, T_max2] -> [B, T_max2, T_max]
        return aligner_durations, aligner_soft, aligner_logprob, aligner_mas

    def average_utterance_prosody(self, u_prosody_pred: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        lengths = ((~src_mask) * 1.0).sum(1)
        u_prosody_pred = u_prosody_pred.sum(1, keepdim=True) / lengths.view(-1, 1, 1)
        return u_prosody_pred

    def forward(
        self,
        tokens: torch.Tensor,
        src_lens: torch.Tensor,
        mels: torch.Tensor,
        mel_lens: torch.Tensor,
        pitches: torch.Tensor,
        energies: torch.Tensor,
        attn_priors: torch.Tensor,
        use_ground_truth: bool = True,
        d_vectors: torch.Tensor = None,
        emo_vectors: torch.Tensor = None,
        speaker_idx: torch.Tensor = None,
        lang_idx: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        src_mask = get_mask_from_lengths(src_lens)  # [B, T_src]
        mel_mask = get_mask_from_lengths(mel_lens)  # [B, T_mel]

        # Token embeddings
        token_embeddings = self.src_word_emb(tokens)  # [B, T_src, C_hidden]
        token_embeddings = token_embeddings.masked_fill(src_mask.unsqueeze(-1), 0.0)

        # Alignment network and durations
        aligner_durations, aligner_soft, aligner_logprob, aligner_mas = self._forward_aligner(
            x=token_embeddings,
            y=mels.transpose(1, 2),
            x_mask=~src_mask[:, None],
            y_mask=~mel_mask[:, None],
            attn_priors=attn_priors,
        )
        dr = aligner_durations  # [B, T_en]

        # Embeddings
        # TODO handle other cases
        speaker_embedding = None
        emotion_embedding = None
        if d_vectors is not None:
            speaker_embedding = torch.nn.functional.normalize(d_vectors)

        if emo_vectors is not None:
            emotion_embedding = torch.nn.functional.normalize(emo_vectors)

        # Encoder
        pos_encoding = positional_encoding(
            self.emb_dim,
            max(token_embeddings.shape[1], max(mel_lens)),
            device=token_embeddings.device,
        )
        encoder_outputs = self.encoder(
            token_embeddings,
            src_mask,
            speaker_embedding=speaker_embedding,
            emotion_embedding=emotion_embedding,
            encoding=pos_encoding,
        )

        # Utterance Prosody encoder
        u_prosody_ref = self.u_norm(self.utterance_prosody_encoder(mels=mels, mel_lens=mel_lens))
        u_prosody_pred = self.u_norm(
            self.average_utterance_prosody(
                u_prosody_pred=self.utterance_prosody_predictor(x=encoder_outputs, mask=src_mask),
                src_mask=src_mask,
            )
        )

        if use_ground_truth:
            encoder_outputs = encoder_outputs + self.u_bottle_out(u_prosody_ref)
        else:
            encoder_outputs = encoder_outputs + self.u_bottle_out(u_prosody_pred)

        # Phoneme prosody encoder
        p_prosody_ref = self.p_norm(
            self.phoneme_prosody_encoder(
                x=encoder_outputs, src_mask=src_mask, mels=mels, mel_lens=mel_lens, encoding=pos_encoding
            )
        )
        p_prosody_pred = self.p_norm(self.phoneme_prosody_predictor(x=encoder_outputs, mask=src_mask))

        if use_ground_truth:
            encoder_outputs = encoder_outputs + self.p_bottle_out(p_prosody_ref)
        else:
            encoder_outputs = encoder_outputs + self.p_bottle_out(p_prosody_pred)

        encoder_outputs_res = encoder_outputs

        # Pitch predictor
        pitch_pred, avg_pitch_target, pitch_emb = self.pitch_adaptor.get_pitch_embedding_train(
            x=encoder_outputs,
            target=pitches,
            dr=dr,
            mask=src_mask,
        )

        # Energy predictor
        energy_pred, avg_energy_target, energy_emb = self.energy_adaptor.get_energy_embedding_train(
            x=encoder_outputs,
            target=energies,
            dr=dr,
            mask=src_mask,
        )

        encoder_outputs = (
            encoder_outputs.transpose(1, 2) + pitch_emb + energy_emb
        )  # [B, T_src, C_hidden] --> [B, C_hidden, T_src]

        # Duration predictor
        # log_duration_prediction = self.duration_predictor(txt_enc=encoder_outputs_res.detach(), lens=src_lens)
        log_duration_prediction = self.duration_predictor(x=encoder_outputs_res.detach(), mask=src_mask)
        # duration_ranges = self.range_predictor(x=encoder_outputs_res, dr=dr, x_lengths=src_lens)

        # Upsample encoder outputs
        # mel_pred_mask, encoder_outputs_ex, alignments = self._expand_encoder_with_gaussian_upsampling(
        #     o_en=encoder_outputs,
        #     o_ranges=duration_ranges.squeeze(2),
        #     y_lengths=mel_lens,
        #     dr=dr,
        #     x_mask=~src_mask[:, None],
        #     x_lengths=src_lens,
        # )
        mel_pred_mask, encoder_outputs_ex, alignments = self._expand_encoder_with_durations(
            o_en=encoder_outputs, y_lengths=mel_lens, dr=dr, x_mask=~src_mask[:, None]
        )

        # Decoder
        x = self.decoder(
            encoder_outputs_ex.transpose(1, 2),
            mel_mask,
            speaker_embedding=speaker_embedding,
            emotion_embedding=emotion_embedding,
            encoding=pos_encoding,
        )
        x = self.to_mel(x)

        dr = torch.log(dr + 1)

        dr_pred = torch.exp(log_duration_prediction) - 1
        alignments_dp = self.generate_attn(dr_pred, src_mask.unsqueeze(1), mel_pred_mask)  # [B, T_max, T_max2']

        return {
            "model_outputs": x,
            "pitch_pred": pitch_pred,
            "pitch_target": avg_pitch_target,
            "energy_pred": energy_pred,
            "energy_target": avg_energy_target,
            "u_prosody_pred": u_prosody_pred,
            "u_prosody_ref": u_prosody_ref,
            "p_prosody_pred": p_prosody_pred,
            "p_prosody_ref": p_prosody_ref,
            "alignments_dp": alignments_dp,
            "alignments": alignments,  # [B, T_de, T_en]
            "aligner_soft": aligner_soft,
            "aligner_mas": aligner_mas,
            "aligner_durations": aligner_durations,
            "aligner_logprob": aligner_logprob,
            "dr_log_pred": log_duration_prediction.squeeze(1),  # [B, T]
            "dr_log_target": dr.squeeze(1),  # [B, T]
            "spk_emb": speaker_embedding,
            "emo_emb": emotion_embedding,
        }

    @torch.no_grad()
    def inference(
        self,
        tokens: torch.Tensor,
        speaker_idx: torch.Tensor,  # TODO
        p_control: float = None,  # TODO
        d_control: float = None,  # TODO
        d_vectors: torch.Tensor = None,
        emo_vectors: torch.Tensor = None,
        pitch_transform: Callable = None,
        energy_transform: Callable = None,
    ) -> torch.Tensor:
        src_mask = get_mask_from_lengths(torch.tensor([tokens.shape[1]], dtype=torch.int64, device=tokens.device))
        src_lens = torch.tensor(tokens.shape[1:2]).to(tokens.device)

        # Token embeddings
        token_embeddings = self.src_word_emb(tokens)
        token_embeddings = token_embeddings.masked_fill(src_mask.unsqueeze(-1), 0.0)

        # Embeddings
        # TODO handle other cases
        speaker_embedding = None
        emotion_embedding = None
        if d_vectors is not None:
            speaker_embedding = torch.nn.functional.normalize(d_vectors)

        if emo_vectors is not None:
            emotion_embedding = torch.nn.functional.normalize(emo_vectors)

        # Encoder
        pos_encoding = positional_encoding(
            self.emb_dim,
            token_embeddings.shape[1],
            device=token_embeddings.device,
        )
        encoder_outputs = self.encoder(
            token_embeddings,
            src_mask,
            speaker_embedding=speaker_embedding,
            emotion_embedding=emotion_embedding,
            encoding=pos_encoding,
        )

        # Utterance Prosody encoder
        u_prosody_pred = self.u_norm(
            self.average_utterance_prosody(
                u_prosody_pred=self.utterance_prosody_predictor(x=encoder_outputs, mask=src_mask),
                src_mask=src_mask,
            )
        )
        encoder_outputs = encoder_outputs + self.u_bottle_out(u_prosody_pred).expand_as(encoder_outputs)

        # Phoneme prosody encoder
        p_prosody_pred = self.p_norm(
            self.phoneme_prosody_predictor(
                x=encoder_outputs,
                mask=src_mask,
            )
        )
        encoder_outputs = encoder_outputs + self.p_bottle_out(p_prosody_pred).expand_as(encoder_outputs)

        encoder_outputs_res = encoder_outputs

        # Pitch predictor
        pitch_emb_pred, pitch_pred = self.pitch_adaptor.get_pitch_embedding(
            x=encoder_outputs,
            mask=src_mask,
            pitch_transform=pitch_transform,
            pitch_mean=self.pitch_mean,
            pitch_std=self.pitch_std,
        )

        # Energy predictor
        energy_emb_pred, energy_pred = self.energy_adaptor.get_energy_embedding(
            x=encoder_outputs, mask=src_mask, energy_transform=energy_transform
        )
        encoder_outputs = encoder_outputs.transpose(1, 2) + pitch_emb_pred + energy_emb_pred

        # Duration predictor
        log_duration_pred = self.duration_predictor(
            x=encoder_outputs_res.detach(), mask=src_mask
        )  # [B, C_hidden, T_src] -> [B, T_src]
        duration_pred = (torch.exp(log_duration_pred) - 1) * (~src_mask) * self.length_scale  # -> [B, T_src]
        duration_pred[duration_pred < 1] = 1.0  # -> [B, T_src]
        duration_pred = torch.round(duration_pred)  # -> [B, T_src]
        mel_lens = duration_pred.sum(1)  # -> [B,]
        # duration_ranges = self.range_predictor(x=encoder_outputs_res, dr=duration_pred.squeeze(1), x_lengths=src_lens)

        # Upsample encoder outputs
        # _, encoder_outputs_ex, alignments = self._expand_encoder_with_gaussian_upsampling(
        #     o_en=encoder_outputs,
        #     o_ranges=duration_ranges.squeeze(2),
        #     y_lengths=mel_lens,
        #     dr=duration_pred.squeeze(1),
        #     x_mask=~src_mask[:None],
        #     x_lengths=src_lens,
        # )

        _, encoder_outputs_ex, alignments = self._expand_encoder_with_durations(
            o_en=encoder_outputs, y_lengths=mel_lens, dr=duration_pred.squeeze(1), x_mask=~src_mask[:, None]
        )

        mel_mask = get_mask_from_lengths(
            torch.tensor([encoder_outputs_ex.shape[2]], dtype=torch.int64, device=encoder_outputs_ex.device)
        )

        if encoder_outputs_ex.shape[1] > pos_encoding.shape[1]:
            encoding = positional_encoding(self.emb_dim, encoder_outputs_ex.shape[2], device=tokens.device)

        # [B, C_hidden, T_src], [B, 1, T_src], [B, C_emb], [B, T_src, C_hidden] -> [B, C_hidden, T_src]
        x = self.decoder(
            encoder_outputs_ex.transpose(1, 2),
            mel_mask,
            speaker_embedding=speaker_embedding,
            emotion_embedding=emotion_embedding,
            encoding=encoding,
        )
        x = self.to_mel(x)
        outputs = {
            "model_outputs": x,
            "alignments": alignments,
            # "pitch": pitch_emb_pred,
            "durations": duration_pred,
            "pitch": pitch_pred,
            "energy": energy_pred,
        }
        return outputs
