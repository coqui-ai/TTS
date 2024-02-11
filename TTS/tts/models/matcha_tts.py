import torch

from TTS.tts.configs.matcha_tts import MatchaTTSConfig
from TTS.tts.models.base_tts import BaseTTS
from TTS.tts.utils.text.tokenizer import TTSTokenizer


class MatchaTTS(BaseTTS):

    def __init__(
        self,
        config: MatchaTTSConfig,
        ap: "AudioProcessor" = None,
        tokenizer: "TTSTokenizer" = None,
    ):
        super().__init__(config, ap, tokenizer)

    def forward(self):
        pass

    @torch.no_grad()
    def inference(self):
        pass

    @staticmethod
    def init_from_config(config: "MatchaTTSConfig"):
        pass

    def load_checkpoint(self, checkpoint_path):
        pass
