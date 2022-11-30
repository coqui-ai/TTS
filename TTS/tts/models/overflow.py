import torch
import torch.nn as nn

from TTS.tts.layers.glow_tts.decoder import Decoder
from TTS.tts.layers.neural_hmm.common_layers import Encoder
from TTS.tts.layers.neural_hmm.hmm import HMM
from TTS.tts.models.base_tts import BaseTTS
from TTS.tts.utils.speakers import SpeakerManager
from TTS.tts.utils.text.tokenizer import TTSTokenizer


class OverFlow(BaseTTS):
    """OverFlow TTS model.
    
    Paper::
        https://arxiv.org/abs/2211.06892
        
    Paper abstract::
        Neural HMMs are a type of neural transducer recently proposed for
    sequence-to-sequence modelling in text-to-speech. They combine the best features
    of classic statistical speech synthesis and modern neural TTS, requiring less
    data and fewer training updates, and are less prone to gibberish output caused
    by neural attention failures. In this paper, we combine neural HMM TTS with
    normalising flows for describing the highly non-Gaussian distribution of speech
    acoustics. The result is a powerful, fully probabilistic model of durations and
    acoustics that can be trained using exact maximum likelihood. Compared to
    dominant flow-based acoustic models, our approach integrates autoregression for
    improved modelling of long-range dependences such as utterance-level prosody.
    Experiments show that a system based on our proposal gives more accurate
    pronunciations and better subjective speech quality than comparable methods,
    whilst retaining the original advantages of neural HMMs. Audio examples and code
    are available at https://shivammehta25.github.io/OverFlow/.
    
    Check :class:`TTS.tts.configs.overflow.OverFlowConfig` for class arguments.
    """
    
    def __init__(
        self, config: "OverFlowConfig",
        ap: "AudioProcessor" = None,
        tokenizer: "TTSTokenizer" = None,
        speaker_manager: SpeakerManager = None,        
    ):
        super().__init__(config, ap, tokenizer, speaker_manager)
        
        # pass all config fields to `self`
        # for fewer code change
        self.config = config
        for key in config:
            setattr(self, key, config[key])
    
        self.decoder_output_dim = config.out_channels
        
        self.encoder = Encoder(self.num_char,config.state_per_phone, config.encoder_in_features)
        self.hmm = HMM(
            self.out_channels,
            self.ar_order,
            self.encoder_dim,
            self.prenet_type,
            self.prenet_dim,
            self.prenet_dropout,
            self.memory_rnn_dim,
            self.prenet_dropout_at_inference,
            self.outputnet_size,
            self.flat_start_params,
            self.std_floor
        )
        
        self.decoder = Decoder(
            self.out_channels,
            self.hidden_channels_dec,
            self.kernel_size_dec,
            self.dilation_rate,
            self.num_flow_blocks_dec,
            self.num_block_layers,
            dropout_p=self.dropout_p_dec,
            num_splits=self.num_splits,
            num_squeeze=self.num_squeeze,
            sigmoid_scale=self.sigmoid_scale,
            c_in_channels=self.c_in_channels
        )
    
    
    def forward(
        self, text, text_len, mels, mel_len
    ):
        """
        Forward pass for training and computing the log likelihood of a given batch.

        Shapes:
            Shapes:
            text: :math:`[B, T_in]`
            text_lengths: :math:`[B]`
            mel_specs: :math:`[B, T_out, C]`
            mel_lengths: :math:`[B]`
        """
        outputs = {
            "log_alpha": None
        }
        
        encoder_outputs, text_lengths = self.encoder(text, text_lengths)
        z, z_lengths, log_det = self.decoder(mels, mel_len)
        log_probs = self.hmm(encoder_outputs, text_lengths, z, z_lengths)

        
        
        