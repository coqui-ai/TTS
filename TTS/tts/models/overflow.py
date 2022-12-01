import os
from typing import Dict, List, Union

import torch
import torch.nn as nn
from coqpit import Coqpit
from tqdm.auto import tqdm

from TTS.tts.configs.overflow_config import OverFlowConfig
from TTS.tts.layers.glow_tts.decoder import Decoder
from TTS.tts.layers.neural_hmm.common_layers import Encoder, OverFlowUtils
from TTS.tts.layers.neural_hmm.hmm import HMM
from TTS.tts.models.base_tts import BaseTTS
from TTS.tts.utils.speakers import SpeakerManager
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.io import load_fsspec


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
        
        self.encoder = Encoder(config.num_chars, config.state_per_phone, config.encoder_in_out_features)
        self.hmm = HMM(
            frame_channels=self.out_channels,
            ar_order=self.ar_order,
            sampling_temp=self.sampling_temp,
            deterministic_transition=self.deterministic_transition,
            duration_threshold=self.duration_threshold,
            encoder_dim=self.encoder_in_out_features,
            prenet_type=self.prenet_type,
            prenet_dim=self.prenet_dim,
            prenet_n_layers=self.prenet_n_layers,
            prenet_dropout=self.prenet_dropout,
            prenet_dropout_at_inference=self.prenet_dropout_at_inference,
            memory_rnn_dim=self.memory_rnn_dim,
            outputnet_size=self.outputnet_size,
            flat_start_params=self.flat_start_params,
            std_floor=self.std_floor,
            use_grad_checkpointing=self.use_grad_checkpointing
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
        
        self.normalizer = None
    
    
    def forward(
        self, text, text_len, mels, mel_len
    ):
        """
        Forward pass for training and computing the log likelihood of a given batch.

        Shapes:
            Shapes:
            text: :math:`[B, T_in]`
            text_len: :math:`[B]`
            mel_specs: :math:`[B, T_out, C]`
            mel_lengths: :math:`[B]`
        """
        if self.normalizer is None:
            pass
            
        
        encoder_outputs, text_len = self.encoder(text, text_len)
        z, z_lengths, logdet = self.decoder(mels, mel_len)
        log_probs, alignments, transition_vectors, means = self.hmm(encoder_outputs, text_len, z, z_lengths, logdet, z_lengths)
        
        outputs = {
            "log_probs": log_probs,
            "alignments": alignments,
            "transition_vectors": transition_vectors,
            "means": means,
        }
        
        return outputs
        
    def train_step(self, batch: dict, criterion: nn.Module):
        text_input = batch["text_input"]
        text_lengths = batch["text_lengths"]
        mel_input = batch["mel_input"]
        mel_lengths = batch["mel_lengths"]
        
        outputs = self.forward(
            text=text_input,
            text_len=text_lengths,
            mels=mel_input,
            mel_len=mel_lengths,
        )
        
        loss_dict = criterion(outputs['log_probs'])
       
        return outputs, loss_dict 
    

    def eval_step(self, batch: Dict, criterion: nn.Module):
        return self.train_step(batch, criterion)
    
    
    def inference(input: torch.Tensor, aux_inputs={}) -> Dict:
        outputs_dict = {"model_outputs": None}
        return outputs_dict
    
    @staticmethod
    def get_criterion():
        from TTS.tts.layers.losses import NLLLoss  # pylint: disable=import-outside-toplevel
        return NLLLoss() 
    
    @staticmethod
    def init_from_config(config: "OverFlowConfig", samples: Union[List[List], List[Dict]] = None, verbose=True):
        """Initiate model from config

        Args:
            config (VitsConfig): Model config.
            samples (Union[List[List], List[Dict]]): Training samples to parse speaker ids for training.
                Defaults to None.
            verbose (bool): If True, print init messages. Defaults to True.
        """
        from TTS.utils.audio import AudioProcessor

        ap = AudioProcessor.init_from_config(config, verbose)
        tokenizer, new_config = TTSTokenizer.init_from_config(config)
        speaker_manager = SpeakerManager.init_from_config(config, samples)
        return OverFlow(new_config, ap, tokenizer, speaker_manager)

    
    def load_checkpoint(self, config: Coqpit, checkpoint_path: str, eval: bool = False, strict: bool = True, cache=False) -> None:
        state = load_fsspec(checkpoint_path, map_location=torch.device("cpu"))
        self.load_state_dict(state["model"])
        if eval:
            self.eval()
            self.store_inverse()
            assert not self.training
    
    def on_init_start(self, trainer):
        if not os.path.isfile(trainer.config.normalized_mel_parameter_path):
            dataloader = trainer.get_train_dataloader(training_assets=None, samples=trainer.train_samples, verbose=False)
            print(f" | > Data parameters not found for: {trainer.config.normalized_mel_parameter_path}. Computing mel normalization parameters...")
            data_mean, data_std, init_transition_prob = OverFlowUtils.get_data_parameters_for_flat_start(dataloader, trainer.config.out_channels, trainer.config.states_per_phone)
            print(f" | > Saving data parameters to: {trainer.config.normalized_mel_parameter_path}: value: {data_mean, data_std, init_transition_prob}")
            torch.save({
                'mean': data_mean,
                'std': data_std,
                'init_transition_prob': init_transition_prob
            }, trainer.config.normalized_mel_parameter_path)
        else:
            print(f" | > Data parameters found for: {trainer.config.normalized_mel_parameter_path}. Loading mel normalization parameters...")
        
        trainer.config.flat_start_params['transition_p'] = init_transition_prob
            
            
            
       
        
        
        