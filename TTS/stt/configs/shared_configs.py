from dataclasses import dataclass, field
from typing import Dict, List

from TTS.config import BaseAudioConfig, BaseDatasetConfig, BaseTrainingConfig


@dataclass
class BaseSTTConfig(BaseTrainingConfig):
    """Shared parameters among all the tts models.

    Args:
        model_type (str):
            The type of the model that defines the trainer mode. Do not change this value.

        audio (BaseAudioConfig):
            Audio processor config object instance.

        enable_eos_bos_chars (bool):
            enable / disable the use of eos and bos characters.

        vocabulary (Dict[str, int]):
            vocabulary used by the model

        batch_group_size (int):
            Size of the batch groups used for bucketing. By default, the dataloader orders samples by the sequence
            length for a more efficient and stable training. If `batch_group_size > 1` then it performs bucketing to
            prevent using the same batches for each epoch.

        loss_masking (bool):
            enable / disable masking loss values against padded segments of samples in a batch.

        feature_extractor (str):
            Name of the feature extractor used by the model, one of the supperted values by
            ```TTS.stt.datasets.dataset.FeatureExtractor```. Defaults to None.

        sort_by_audio_len (bool):
            If true, dataloder sorts the data by audio length else sorts by the input text length. Defaults to `False`.

        min_seq_len (int):
            Minimum sequence length to be used at training.

        max_seq_len (int):
            Maximum sequence length to be used at training. Larger values result in more VRAM usage.

        train_datasets (List[BaseDatasetConfig]):
            List of datasets used for training. If multiple datasets are provided, they are merged and used together
            for training.

        eval_datasets (List[BaseDatasetConfig]):
            List of datasets used for evaluation. If multiple datasets are provided, they are merged and used together
            for training.

        optimizer (str):
            Optimizer used for the training. Set one from `torch.optim.Optimizer` or `TTS.utils.training`.
            Defaults to ``.

        optimizer_params (dict):
            Optimizer kwargs. Defaults to `{"betas": [0.8, 0.99], "weight_decay": 0.0}`

        lr_scheduler (str):
            Learning rate scheduler for the training. Use one from `torch.optim.Scheduler` schedulers or
            `TTS.utils.training`. Defaults to ``.

        lr_scheduler_params (dict):
            Parameters for the generator learning rate scheduler. Defaults to `{"warmup": 4000}`.
    """

    model_type: str = "stt"
    audio: BaseAudioConfig = field(default_factory=BaseAudioConfig)
    # phoneme settings
    enable_eos_bos_chars: bool = False
    # vocabulary parameters
    vocabulary: dict = None
    # training params
    batch_group_size: int = 0
    loss_masking: bool = None
    # dataloading
    feature_extractor: str = None
    sort_by_audio_len: bool = True
    min_seq_len: int = 1
    max_seq_len: int = float("inf")
    # dataset
    train_datasets: List[BaseDatasetConfig] = field(default_factory=lambda: [BaseDatasetConfig()])
    eval_datasets: List[BaseDatasetConfig] = field(default_factory=lambda: [BaseDatasetConfig()])
    # optimizer
    optimizer: str = None
    optimizer_params: dict = None
    # scheduler
    lr_scheduler: str = ""
    lr_scheduler_params: dict = None
