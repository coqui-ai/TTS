import math

import torch
from torch import nn
from torch.nn import functional as F

from TTS.tts.layers.glow_tts.decoder import Decoder
from TTS.tts.layers.glow_tts.encoder import Encoder
from TTS.tts.layers.glow_tts.monotonic_align import generate_path, maximum_path
from TTS.tts.utils.generic_utils import sequence_mask
from TTS.tts.layers.tacotron.adverserial_classifier import ReversalClassifier
from TTS.tts.layers.glow_tts.duration_predictor import DeterministicDurationPredictor, StochasticDurationPredictor

def average_pitch(pitch, durs):
    durs_cums_ends = torch.cumsum(durs, dim=1).long()
    durs_cums_starts = torch.nn.functional.pad(durs_cums_ends[:, :-1], (1, 0))
    pitch_nonzero_cums = torch.nn.functional.pad(torch.cumsum(pitch != 0.0, dim=2), (1, 0))
    pitch_cums = torch.nn.functional.pad(torch.cumsum(pitch, dim=2), (1, 0))

    bs, l = durs_cums_ends.size()
    n_formants = pitch.size(1)
    dcs = durs_cums_starts[:, None, :].expand(bs, n_formants, l)
    dce = durs_cums_ends[:, None, :].expand(bs, n_formants, l)

    pitch_sums = (torch.gather(pitch_cums, 2, dce) - torch.gather(pitch_cums, 2, dcs)).float()
    pitch_nelems = (torch.gather(pitch_nonzero_cums, 2, dce) - torch.gather(pitch_nonzero_cums, 2, dcs)).float()

    pitch_avg = torch.where(pitch_nelems == 0.0, pitch_nelems, pitch_sums / pitch_nelems)

    return pitch_avg

class GlowTTS(nn.Module):
    """Glow TTS models from https://arxiv.org/abs/2005.11129

    Args:
        num_chars (int): number of embedding characters.
        hidden_channels_enc (int): number of embedding and encoder channels.
        hidden_channels_dec (int): number of decoder channels.
        use_encoder_prenet (bool): enable/disable prenet for encoder. Prenet modules are hard-coded for each alternative encoder.
        hidden_channels_dp (int): number of duration predictor channels.
        out_channels (int): number of output channels. It should be equal to the number of spectrogram filter.
        num_flow_blocks_dec (int): number of decoder blocks.
        kernel_size_dec (int): decoder kernel size.
        dilation_rate (int): rate to increase dilation by each layer in a decoder block.
        num_block_layers (int): number of decoder layers in each decoder block.
        dropout_p_dec (float): dropout rate for decoder.
        num_speaker (int): number of speaker to define the size of speaker embedding layer.
        c_in_channels (int): number of speaker embedding channels. It is set to 512 if embeddings are learned.
        num_splits (int): number of split levels in inversible conv1x1 operation.
        num_squeeze (int): number of squeeze levels. When squeezing channels increases and time steps reduces by the factor 'num_squeeze'.
        sigmoid_scale (bool): enable/disable sigmoid scaling in decoder.
        mean_only (bool): if True, encoder only computes mean value and uses constant variance for each time step.
        encoder_type (str): encoder module type.
        encoder_params (dict): encoder module parameters.
        external_speaker_embedding_dim (int): channels of external speaker embedding vectors.
    """

    def __init__(
        self,
        num_chars,
        hidden_channels_enc,
        hidden_channels_dec,
        use_encoder_prenet,
        hidden_channels_dp,
        out_channels,
        num_langs=1,
        language_embedding_dim=0,
        num_flow_blocks_dec=12,
        kernel_size_dec=5,
        dilation_rate=5,
        num_block_layers=4,
        dropout_p_dp=0.1,
        dropout_p_dec=0.05,
        num_speakers=0,
        c_in_channels=0,
        num_splits=4,
        num_squeeze=1,
        sigmoid_scale=False,
        mean_only=False,
        encoder_type="transformer",
        encoder_params=None,
        external_speaker_embedding_dim=None,
        use_stochastic_dp=False,
        dp_use_language_embedding=False,
        reversal_classifier=False,
        reversal_classifier_dim=256,
        reversal_gradient_clipping=0.25,
        use_pitch_predictor=False,
        pitch_predictor_use_language_embedding=False,
        pitch_predictor_hidden_channels=256,
        pitch_predictor_dropout=0.1,
        pitch_predictor_kernel_size=3,
        pitch_predictor_dropout_p=0.1,
        pitch_embedding_kernel_size=3,
    ):

        super().__init__()
        self.num_chars = num_chars
        self.hidden_channels_dp = hidden_channels_dp
        self.hidden_channels_enc = hidden_channels_enc
        self.hidden_channels_dec = hidden_channels_dec
        self.out_channels = out_channels
        self.num_flow_blocks_dec = num_flow_blocks_dec
        self.kernel_size_dec = kernel_size_dec
        self.dilation_rate = dilation_rate
        self.num_block_layers = num_block_layers
        self.dropout_p_dec = dropout_p_dec
        self.num_speakers = num_speakers
        self.c_in_channels = c_in_channels
        self.num_splits = num_splits
        self.num_squeeze = num_squeeze
        self.sigmoid_scale = sigmoid_scale
        self.mean_only = mean_only
        self.use_encoder_prenet = use_encoder_prenet
        self.use_stochastic_dp = use_stochastic_dp
        self.use_pitch_predictor = use_pitch_predictor

        # model constants.
        self.noise_scale = 0.0  # defines the noise variance applied to the random z vector at inference.
        self.length_scale = 1.0  # scaler for the duration predictor. The larger it is, the slower the speech.
        self.noise_scale_w = 1.0 # defines the noise variance applied to the duration predictor z vector at inference.

        self.pitch_transform = None
        self.pitch_mean = None
        self.pitch_std = None

        self.pitch_transform_amplify_factor = 1.0
        self.pitch_transform_shift_factor = 0.0

        self.external_speaker_embedding_dim = external_speaker_embedding_dim

        self.reversal_classifier = reversal_classifier

        # if is a multispeaker and c_in_channels is 0, set to 512
        if num_speakers > 1:
            if self.c_in_channels == 0 and not self.external_speaker_embedding_dim:
                self.c_in_channels = 512
            elif self.external_speaker_embedding_dim:
                self.c_in_channels = self.external_speaker_embedding_dim

        if num_langs > 1:
            if language_embedding_dim is None: 
                language_embedding_dim = num_langs if num_langs % 2 == 0 else num_langs + 1 # Allow for odd number of languages

        self.encoder = Encoder(
            num_chars,
            out_channels=out_channels,
            hidden_channels=hidden_channels_enc,
            hidden_channels_dp=hidden_channels_dp,
            encoder_type=encoder_type,
            encoder_params=encoder_params,
            mean_only=mean_only,
            use_prenet=use_encoder_prenet,
            dropout_p_dp=dropout_p_dp,
            num_langs=num_langs,
            language_embedding_dim=language_embedding_dim
        )

        self.decoder = Decoder(
            out_channels,
            hidden_channels_dec,
            kernel_size_dec,
            dilation_rate,
            num_flow_blocks_dec,
            num_block_layers,
            dropout_p=dropout_p_dec,
            num_splits=num_splits,
            num_squeeze=num_squeeze,
            sigmoid_scale=sigmoid_scale,
            c_in_channels=self.c_in_channels,
        )
        if self.use_stochastic_dp:
            self.duration_predictor = StochasticDurationPredictor(
                hidden_channels_enc + language_embedding_dim, hidden_channels_dp, 3, g_channels=self.c_in_channels, 
                language_embedding_dim=language_embedding_dim, use_language_embedding=dp_use_language_embedding,
            )
        else:
            self.duration_predictor = DeterministicDurationPredictor(
                    hidden_channels_enc + self.c_in_channels + language_embedding_dim, hidden_channels_dp, 3, dropout_p_dp, 
                    language_embedding_dim=language_embedding_dim, use_language_embedding=dp_use_language_embedding,
                )

        if self.use_pitch_predictor:
            self.pitch_predictor = DeterministicDurationPredictor(
                    out_channels, pitch_predictor_hidden_channels, pitch_predictor_kernel_size, pitch_predictor_dropout, 
                    language_embedding_dim=language_embedding_dim, use_language_embedding=pitch_predictor_use_language_embedding,
                )
            # nn.Sequential(
            self.pitch_emb = nn.Conv1d(
                1,
                out_channels,
                kernel_size=pitch_embedding_kernel_size,
                padding=int((pitch_embedding_kernel_size- 1) / 2),
            )


        if num_speakers > 1 and not external_speaker_embedding_dim:
            # speaker embedding layer
            self.emb_g = nn.Embedding(num_speakers, self.c_in_channels)
            nn.init.uniform_(self.emb_g.weight, -0.1, 0.1)

        # adverserial speaker classifier
        if self.reversal_classifier:
            self._reversal_classifier = ReversalClassifier(
                input_dim=out_channels,
                hidden_dim=reversal_classifier_dim,
                output_dim=num_speakers,
                gradient_clipping_bounds=reversal_gradient_clipping,
            )

    @staticmethod
    def compute_outputs(attn, o_mean, o_log_scale, x_mask):
        # compute final values with the computed alignment
        y_mean = torch.matmul(attn.squeeze(1).transpose(1, 2), o_mean.transpose(1, 2)).transpose(
            1, 2
        )  # [b, t', t], [b, t, d] -> [b, d, t']
        y_log_scale = torch.matmul(attn.squeeze(1).transpose(1, 2), o_log_scale.transpose(1, 2)).transpose(
            1, 2
        )  # [b, t', t], [b, t, d] -> [b, d, t']
        # compute total duration with adjustment
        o_attn_dur = torch.log(1 + torch.sum(attn, -1)) * x_mask
        return y_mean, y_log_scale, o_attn_dur

    def _forward_pitch_predictor(self, o_en, x_mask, pitch=None, dr=None, language_embedding=None):
        o_pitch = self.pitch_predictor(o_en, x_mask, language_embedding=language_embedding)
        if pitch is not None:
            avg_pitch = average_pitch(pitch, dr)
            o_pitch_emb = self.pitch_emb(avg_pitch)
            return o_pitch_emb, o_pitch, avg_pitch

        if self.pitch_transform is not None:
            o_pitch = self.apply_pitch_trans(o_pitch, self.pitch_transform)

        o_pitch_emb = self.pitch_emb(o_pitch)
        return o_pitch_emb, o_pitch, None

    def apply_pitch_trans(self, pitch, trans):
        if trans == 'flatten':
            pitch = pitch * 0.0
        elif trans == 'invert':
            pitch = pitch * -1.0
        elif trans == 'amplify':
            pitch = pitch * self.pitch_transform_amplify_factor 
        elif trans == 'shift':
            pitch = (pitch + self.pitch_transform_shift_factor) / self.pitch_std
        else:
            raise RuntimeError("Invalid Pitch Tranform: {}".format(trans))
        return pitch

    def forward(self, x, x_lengths, y=None, y_lengths=None, attn=None, g=None, language_ids=None, pitch=None):
        """
        Shapes:
            x: [B, T]
            x_lenghts: B
            y: [B, C, T]
            y_lengths: B
            g: [B, C] or B
        """
        y_max_length = y.size(2)
        # norm speaker embeddings
        if g is not None:
            if self.external_speaker_embedding_dim:
                g = F.normalize(g).unsqueeze(-1)
            else:
                g = F.normalize(self.emb_g(g)).unsqueeze(-1)  # [b, h, 1]

        # embedding pass
        o_mean, o_log_scale, x_dp, x_mask, language_embedding = self.encoder(x, x_lengths, language_ids=language_ids)
        # drop redisual frames wrt num_squeeze and set y_lengths.
        y, y_lengths, y_max_length, attn = self.preprocess(y, y_lengths, y_max_length, None)
        # create masks
        y_mask = torch.unsqueeze(sequence_mask(y_lengths, y_max_length), 1).to(x_mask.dtype)
        attn_mask = torch.unsqueeze(x_mask, -1) * torch.unsqueeze(y_mask, 2)
        # decoder pass
        z, logdet = self.decoder(y, y_mask, g=g, reverse=False)

        speaker_prediction = self._reversal_classifier(z.transpose(1, 2)) if self.reversal_classifier else None

        # find the alignment path
        with torch.no_grad():
            o_scale = torch.exp(-2 * o_log_scale)
            logp1 = torch.sum(-0.5 * math.log(2 * math.pi) - o_log_scale, [1]).unsqueeze(-1)  # [b, t, 1]
            logp2 = torch.matmul(o_scale.transpose(1, 2), -0.5 * (z ** 2))  # [b, t, d] x [b, d, t'] = [b, t, t']
            logp3 = torch.matmul((o_mean * o_scale).transpose(1, 2), z)  # [b, t, d] x [b, d, t'] = [b, t, t']
            logp4 = torch.sum(-0.5 * (o_mean ** 2) * o_scale, [1]).unsqueeze(-1)  # [b, t, 1]
            logp = logp1 + logp2 + logp3 + logp4  # [b, t, t']
            attn = maximum_path(logp, attn_mask.squeeze(1)).unsqueeze(1).detach()

        attn_dur = torch.sum(attn, -1)

        # duration predictor
        if self.use_stochastic_dp:
            logw = self.duration_predictor(x_dp, x_mask, attn_dur, g=g, language_embedding=language_embedding)
            logw = logw / torch.sum(x_mask)
        else:
            logw = self.duration_predictor(x_dp, x_mask, g=g, language_embedding=language_embedding)

        if self.use_pitch_predictor:
            # add pitch
            o_pitch_emb, o_pitch, avg_pitch = self._forward_pitch_predictor(o_mean, x_mask, pitch=pitch, dr=attn_dur.squeeze(), language_embedding=language_embedding)
            o_mean += o_pitch_emb
        else:
            o_pitch, avg_pitch = None, None
        y_mean, y_log_scale, o_attn_dur = self.compute_outputs(attn, o_mean, o_log_scale, x_mask)


        attn = attn.squeeze(1).permute(0, 2, 1)
        return z, logdet, y_mean, y_log_scale, attn, logw, o_attn_dur, speaker_prediction, o_pitch, avg_pitch
    
    @torch.no_grad()
    def inference(self, x, x_lengths, g=None, language_ids=None, pitch=None):
        if g is not None:
            if self.external_speaker_embedding_dim:
                g = F.normalize(g).unsqueeze(-1)
            else:
                g = F.normalize(self.emb_g(g)).unsqueeze(-1)  # [b, h]

        # embedding pass
        o_mean, o_log_scale, x_dp, x_mask, language_embedding = self.encoder(x, x_lengths, language_ids=language_ids)

        # duration predictor
        if self.use_stochastic_dp:
            # reverse flow and predict the duration
            o_dur_log = self.duration_predictor(x_dp, x_mask, g=g, language_embedding=language_embedding, reverse=True, noise_scale=self.noise_scale_w)
            # check if exp here is necessary, during the training the duration preditor 
            w = torch.exp(o_dur_log) * x_mask * self.length_scale
        else:
            o_dur_log = self.duration_predictor(x_dp, x_mask, g=g, language_embedding=language_embedding)
            w = (torch.exp(o_dur_log) - 1) * x_mask * self.length_scale

        # compute output durations
        w_ceil = torch.ceil(w)
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_max_length = None
        # compute masks
        y_mask = torch.unsqueeze(sequence_mask(y_lengths, y_max_length), 1).to(x_mask.dtype)
        attn_mask = torch.unsqueeze(x_mask, -1) * torch.unsqueeze(y_mask, 2)
        # compute attention mask
        attn = generate_path(w_ceil.squeeze(1), attn_mask.squeeze(1)).unsqueeze(1)

        if self.use_pitch_predictor:
            # add pitch
            o_pitch_emb, _, _ = self._forward_pitch_predictor(o_mean, x_mask, pitch=pitch, dr=torch.sum(attn, -1).squeeze(), language_embedding=language_embedding)
            o_mean += o_pitch_emb

        y_mean, y_log_scale, o_attn_dur = self.compute_outputs(attn, o_mean, o_log_scale, x_mask)

        z = (y_mean + torch.exp(y_log_scale) * torch.randn_like(y_mean) * self.noise_scale) * y_mask
        # decoder pass
        y, logdet = self.decoder(z, y_mask, g=g, reverse=True)
        attn = attn.squeeze(1).permute(0, 2, 1)
        return y, logdet, y_mean, y_log_scale, attn, o_dur_log, o_attn_dur

    @torch.no_grad()
    def inference_with_MAS(self, x, x_lengths, y=None, y_lengths=None, attn=None, g=None, language_ids=None, pitch=None):
        """
        It's similar to the teacher forcing in Tacotron.
        It was proposed in: https://arxiv.org/abs/2104.05557
        Shapes:
            x: [B, T]
            x_lenghts: B
            y: [B, C, T]
            y_lengths: B
            g: [B, C] or B
        """
        y_max_length = y.size(2)
        # norm speaker embeddings
        if g is not None:
            if self.external_speaker_embedding_dim:
                g = F.normalize(g).unsqueeze(-1)
            else:
                g = F.normalize(self.emb_g(g)).unsqueeze(-1)  # [b, h, 1]

        # embedding pass
        o_mean, o_log_scale, _, x_mask, language_embedding= self.encoder(x, x_lengths, language_ids=language_ids)
        # drop redisual frames wrt num_squeeze and set y_lengths.
        y, y_lengths, y_max_length, attn = self.preprocess(y, y_lengths, y_max_length, None)
        # create masks
        y_mask = torch.unsqueeze(sequence_mask(y_lengths, y_max_length), 1).to(x_mask.dtype)
        attn_mask = torch.unsqueeze(x_mask, -1) * torch.unsqueeze(y_mask, 2)
        # decoder pass
        z, logdet = self.decoder(y, y_mask, g=g, reverse=False)
        # find the alignment path between z and encoder output
        o_scale = torch.exp(-2 * o_log_scale)
        logp1 = torch.sum(-0.5 * math.log(2 * math.pi) - o_log_scale, [1]).unsqueeze(-1)  # [b, t, 1]
        logp2 = torch.matmul(o_scale.transpose(1, 2), -0.5 * (z ** 2))  # [b, t, d] x [b, d, t'] = [b, t, t']
        logp3 = torch.matmul((o_mean * o_scale).transpose(1, 2), z)  # [b, t, d] x [b, d, t'] = [b, t, t']
        logp4 = torch.sum(-0.5 * (o_mean ** 2) * o_scale, [1]).unsqueeze(-1)  # [b, t, 1]
        logp = logp1 + logp2 + logp3 + logp4  # [b, t, t']
        attn = maximum_path(logp, attn_mask.squeeze(1)).unsqueeze(1).detach()

        if self.use_pitch_predictor:
            # add pitch
            o_pitch_emb, _, _ = self._forward_pitch_predictor(o_mean, x_mask, pitch=pitch, dr=torch.sum(attn, -1).squeeze())
            o_mean += o_pitch_emb

        y_mean, y_log_scale, o_attn_dur = self.compute_outputs(attn, o_mean, o_log_scale, x_mask)
        attn = attn.squeeze(1).permute(0, 2, 1)

        # get predited aligned distribution
        z = y_mean * y_mask

        # reverse the decoder and predict using the aligned distribution
        y, logdet = self.decoder(z, y_mask, g=g, reverse=True)

        return y, logdet, y_mean, y_log_scale, attn, None, o_attn_dur

    @torch.no_grad()
    def decoder_inference(self, y, y_lengths=None, g=None, g_target=None):
        """
        Shapes:
            y: [B, C, T]
            y_lengths: B
            g: [B, C] or B
        """

        y_max_length = y.size(2)
        # norm speaker embeddings
        if g is not None:
            if self.external_speaker_embedding_dim:
                g = F.normalize(g).unsqueeze(-1)
            else:
                g = F.normalize(self.emb_g(g)).unsqueeze(-1)  # [b, h, 1]

        if g_target is not None:
            if self.external_speaker_embedding_dim:
                g_target = F.normalize(g_target).unsqueeze(-1)
            else:
                g_target = F.normalize(self.emb_g(g_target)).unsqueeze(-1)  # [b, h, 1]
        else:
            g_target = g

        y_mask = torch.unsqueeze(sequence_mask(y_lengths, y_max_length), 1).to(y.dtype)

        # decoder pass
        z, logdet = self.decoder(y, y_mask, g=g, reverse=False)
     
        # reverse decoder and predict
        y, logdet = self.decoder(z, y_mask, g=g_target, reverse=True)

        return y, logdet

    def preprocess(self, y, y_lengths, y_max_length, attn=None):
        if y_max_length is not None:
            y_max_length = (y_max_length // self.num_squeeze) * self.num_squeeze
            y = y[:, :, :y_max_length]
            if attn is not None:
                attn = attn[:, :, :, :y_max_length]
        y_lengths = (y_lengths // self.num_squeeze) * self.num_squeeze
        return y, y_lengths, y_max_length, attn

    def store_inverse(self):
        self.decoder.store_inverse()

    def load_checkpoint(
        self, config, checkpoint_path, eval=False
    ):  # pylint: disable=unused-argument, redefined-builtin
        state = torch.load(checkpoint_path, map_location=torch.device("cpu"))
        self.load_state_dict(state["model"])
        if eval:
            self.eval()
            self.store_inverse()
            assert not self.training
