from coqpit import Coqpit
from torch import nn
from TTS.model import BaseTrainerModel
from TTS.tts.layers.generic.wavenet import WNBlocks
from dataclasses import dataclass
from typing import Dict, List, Tuple, Union
from TTS.enhancer.datasets.dataset import EnhancerDataset
import torch.distributed as dist
from torch.utils.data import DataLoader

@dataclass
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

    def init_from_config(config: Coqpit):
        from TTS.utils.audio import AudioProcessor
        ap = AudioProcessor.init_from_config(config)
        return BWE(config, ap)

    def forward(self, x):
        outputs = {
            "y_hat": self.generator(x),
        }
        return outputs

    def inference(self, x):
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

    def get_data_loader(
        self,
        config: Coqpit,
        assets: Dict,
        is_eval: bool,
        samples: Union[List[Dict], List[List]],
        verbose: bool,
        num_gpus: int,
        rank: int = None,
    ) -> "DataLoader":
        if is_eval and not config.run_eval:
            loader = None
        else:
            # init dataloader
            dataset = EnhancerDataset(
                config,
                self.ap,
                samples,
                voice_len=1.6,
                augmentation_config=config.augmentation_config,
                use_torch_spec=None,
            )

            # wait all the DDP process to be ready
            if num_gpus > 1:
                dist.barrier()

            # get samplers
            sampler = self.get_sampler(config, dataset, num_gpus)

            loader = DataLoader(
                dataset,
                batch_size=config.eval_batch_size if is_eval else config.batch_size,
                shuffle=True,  # shuffle is done in the dataset.
                drop_last=False,  # setting this False might cause issues in AMP training.
                sampler=sampler,
                collate_fn=dataset.collate_fn,
                num_workers=config.num_eval_loader_workers if is_eval else config.num_loader_workers,
                pin_memory=False,
            )
        return loader
