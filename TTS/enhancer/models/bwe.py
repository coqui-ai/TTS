from coqpit import Coqpit
from torch import nn
import torch
from TTS.model import BaseTrainerModel
from TTS.tts.layers.generic.wavenet import WNBlocks
from dataclasses import dataclass
from typing import Dict, List, Tuple, Union
from TTS.enhancer.datasets.dataset import EnhancerDataset
import torch.distributed as dist
from torch.utils.data import DataLoader, Sampler
from trainer.torch import DistributedSampler, DistributedSamplerWrapper
from trainer.trainer_utils import get_optimizer, get_scheduler
import torch
from torch.nn import Upsample

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
        self.input_sr = config.input_sr
        self.target_sr = config.target_sr
        self.scale_factor = (self.target_sr / self.input_sr, )

        self.upsample = Upsample(scale_factor=self.scale_factor)
        #self.preconv = nn.Conv1d(1, self.args.num_channel_wn, kernel_size=1)
        self.postconv = nn.Conv1d(self.args.num_channel_wn, 1, kernel_size=1)
        self.generator = WNBlocks(
            in_channels=1,
            hidden_channels=self.args.num_channel_wn,
            kernel_size=1,
            dilation_rate=self.args.dilation_rate_wn,
            num_blocks=self.args.num_blocks_wn,
            num_layers=self.args.num_layers_wn,
        )

    def init_from_config(config: Coqpit):
        from TTS.utils.audio import AudioProcessor
        ap = AudioProcessor.init_from_config(config)
        return BWE(config, ap)

    def forward(self, x):
        x = self.upsample(x)
        x = self.generator(x)
        x = self.postconv(x)
        return {
            "y_hat": x,
        }

    def inference(self, x):
        x = self.upsample(x.unsqueeze(1))
        x = self.generator(x)
        x = self.postconv(x)
        x = x.transpose(1, 2)
        return {
            "y_hat": x,
        }

    def train_step(self, batch: dict, criterion: nn.Module):
        x = batch["input_wav"].unsqueeze(1)
        y = batch["target_wav"].unsqueeze(1)
        lens = batch["target_lens"]
        outputs = self.forward(x)
        loss_dict = criterion(outputs["y_hat"], y, lens)
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
                augmentation_config=config.audio_augmentation,
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

    def load_checkpoint(
        self,
        config,
        checkpoint_path,
        eval=False,
        strict=True,
    ):  # pylint: disable=unused-argument, redefined-builtin
        """Load the model checkpoint and setup for training or inference"""
        state = torch.load(checkpoint_path, map_location=torch.device("cpu"))
        self.load_state_dict(state["model"], strict=strict)
        if eval:
            self.eval()
            assert not self.training

    def get_optimizer(self) -> List:
        return get_optimizer(self.config.optimizer, self.config.optimizer_params, self.config.lr, self)

    def get_lr(self) -> List:
        return self.config.lr

    def get_scheduler(self, optimizer) -> List:
        return get_scheduler(self.config.lr_scheduler, self.config.lr_scheduler_params, optimizer)

    def get_criterion(self):
        from TTS.tts.layers.losses import (  # pylint: disable=import-outside-toplevel
            L1LossMasked
        )
        return L1LossMasked(False)

    def get_sampler(self, config: Coqpit, dataset: EnhancerDataset, num_gpus=1):
        sampler = None
        # sampler for DDP
        if sampler is None:
            sampler = DistributedSampler(dataset) if num_gpus > 1 else None
        else:  # If a sampler is already defined use this sampler and DDP sampler together
            sampler = DistributedSamplerWrapper(sampler) if num_gpus > 1 else sampler
        return sampler
