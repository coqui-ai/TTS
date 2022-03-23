from coqpit import Coqpit
from torch import nn
from TTS.model import BaseTrainerModel
from TTS.tts.layers.generic.wavenet import WNBlocks

class BWEArgs(Coqpit):
    num_channel_wn: int = 128
    dilation_rate_wn: int = 3
    num_blocks_wn: int = 2
    num_layers_wn: int = 7

class BWE(BaseTrainerModel):
    def __init__(
        self,
        config: Coqpit,
        ap: "AudioProcessor",
    ):
        super().__init__()
        self.config = config
        self.ap = ap
        self.args = config.model_args
        self.generator = WNBlocks(
            in_channels=self.args.num_channel_wn,
            hidden_channels=self.args.num_channel_wn,
            kernel_size=1,
            dilation_rate=self.args.dilation_rate_wn,
            num_blocks=self.args.num_blocks_wn,
        )

    def forward(self, x):
        outputs = {
            "y_hat": self.generator(x),
        }
        return outputs

    def train_step(self, batch: dict, criterion: nn.Module):
        x = batch["input_wav"]
        y = batch["target_wav"]
        outputs = self.forward(x)
        loss_dict = criterion(outputs["y_hat"], y)
        return outputs, loss_dict

    def eval_step(self, batch: dict, criterion: nn.Module):
        return self.train_step(batch, criterion)

    
