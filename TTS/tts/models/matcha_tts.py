from dataclasses import field
import math
import torch

from TTS.tts.configs.matcha_tts import MatchaTTSConfig
from TTS.tts.layers.glow_tts.encoder import Encoder
from TTS.tts.layers.matcha_tts.decoder import Decoder
from TTS.tts.models.base_tts import BaseTTS
from TTS.tts.utils.helpers import maximum_path, sequence_mask
from TTS.tts.utils.text.tokenizer import TTSTokenizer


class MatchaTTS(BaseTTS):

    def __init__(
        self,
        config: MatchaTTSConfig,
        ap: "AudioProcessor" = None,
        tokenizer: "TTSTokenizer" = None,
    ):
        super().__init__(config, ap, tokenizer)
        self.encoder = Encoder(
            self.config.num_chars,
            out_channels=80,
            hidden_channels=192,
            hidden_channels_dp=256,
            encoder_type='rel_pos_transformer',
            encoder_params={
                "kernel_size": 3,
                "dropout_p": 0.1,
                "num_layers": 6,
                "num_heads": 2,
                "hidden_channels_ffn": 768,
            }
        )

        self.decoder = Decoder()

    def forward(self, x, x_lengths, y, y_lengths):
        """
        Args:
            x (torch.Tensor):
                Input text sequence ids. :math:`[B, T_en]`

            x_lengths (torch.Tensor):
                Lengths of input text sequences. :math:`[B]`

            y (torch.Tensor):
                Target mel-spectrogram frames. :math:`[B, T_de, C_mel]`

            y_lengths (torch.Tensor):
                Lengths of target mel-spectrogram frames. :math:`[B]`
        """
        y = y.transpose(1, 2)
        y_max_length = y.size(2)

        o_mean, o_log_scale, o_log_dur, o_mask = self.encoder(x, x_lengths, g=None)

        y_mask = torch.unsqueeze(sequence_mask(y_lengths, y_max_length), 1).to(o_mask.dtype)
        attn_mask = torch.unsqueeze(o_mask, -1) * torch.unsqueeze(y_mask, 2)

        with torch.no_grad():
            o_scale = torch.exp(-2 * o_log_scale)
            logp1 = torch.sum(-0.5 * math.log(2 * math.pi) - o_log_scale, [1]).unsqueeze(-1)
            logp2 = torch.matmul(o_scale.transpose(1, 2), -0.5 * (y**2))
            logp3 = torch.matmul((o_mean * o_scale).transpose(1, 2), y)
            logp4 = torch.sum(-0.5 * (o_mean**2) * o_scale, [1]).unsqueeze(-1)
            logp = logp1 + logp2 + logp3 + logp4
            attn = maximum_path(logp, attn_mask.squeeze(1)).unsqueeze(1).detach()

        # Align encoded text with mel-spectrogram and get mu_y segment
        c_mean = torch.matmul(attn.squeeze(1).transpose(1, 2), o_mean.transpose(1, 2)).transpose(1, 2)

        _ = self.decoder(x_1=y, mean=c_mean, mask=y_mask)

    @torch.no_grad()
    def inference(self):
        pass

    @staticmethod
    def init_from_config(config: "MatchaTTSConfig"):
        pass

    def load_checkpoint(self, checkpoint_path):
        pass
