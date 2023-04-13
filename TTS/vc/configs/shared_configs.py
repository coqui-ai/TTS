from dataclasses import asdict, dataclass, field
from typing import Dict, List

from coqpit import Coqpit, check_argument

from TTS.config import BaseAudioConfig, BaseDatasetConfig, BaseTrainingConfig


@dataclass
class BaseVCConfig(BaseTrainingConfig):
    """Shared parameters among all the tts models.

    Args:

        audio (BaseAudioConfig):
            Audio processor config object instance.

        batch_group_size (int):
            Size of the batch groups used for bucketing. By default, the dataloader orders samples by the sequence
            length for a more efficient and stable training. If `batch_group_size > 1` then it performs bucketing to
            prevent using the same batches for each epoch.

        loss_masking (bool):
            enable / disable masking loss values against padded segments of samples in a batch.

        min_text_len (int):
            Minimum length of input text to be used. All shorter samples will be ignored. Defaults to 0.

        max_text_len (int):
            Maximum length of input text to be used. All longer samples will be ignored. Defaults to float("inf").

        min_audio_len (int):
            Minimum length of input audio to be used. All shorter samples will be ignored. Defaults to 0.

        max_audio_len (int):
            Maximum length of input audio to be used. All longer samples will be ignored. The maximum length in the
            dataset defines the VRAM used in the training. Hence, pay attention to this value if you encounter an
            OOM error in training. Defaults to float("inf").

        compute_f0 (int):
            (Not in use yet).

        compute_energy (int):
            (Not in use yet).

        compute_linear_spec (bool):
            If True data loader computes and returns linear spectrograms alongside the other data.

        precompute_num_workers (int):
            Number of workers to precompute features. Defaults to 0.

        use_noise_augment (bool):
            Augment the input audio with random noise.

        start_by_longest (bool):
            If True, the data loader will start loading the longest batch first. It is useful for checking OOM issues.
            Defaults to False.

        shuffle (bool):
            If True, the data loader will shuffle the dataset when there is not sampler defined. Defaults to True.

        drop_last (bool):
            If True, the data loader will drop the last batch if it is not complete. It helps to prevent
            issues that emerge from the partial batch statistics. Defaults to True.

        add_blank (bool):
            Add blank characters between each other two characters. It improves performance for some models at expense
            of slower run-time due to the longer input sequence.

        datasets (List[BaseDatasetConfig]):
            List of datasets used for training. If multiple datasets are provided, they are merged and used together
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

        test_sentences (List[str]):
            List of sentences to be used at testing. Defaults to '[]'

        eval_split_max_size (int):
            Number maximum of samples to be used for evaluation in proportion split. Defaults to None (Disabled).

        eval_split_size (float):
            If between 0.0 and 1.0 represents the proportion of the dataset to include in the evaluation set.
            If > 1, represents the absolute number of evaluation samples. Defaults to 0.01 (1%).

        use_speaker_weighted_sampler (bool):
            Enable / Disable the batch balancer by speaker. Defaults to ```False```.

        speaker_weighted_sampler_alpha (float):
            Number that control the influence of the speaker sampler weights. Defaults to ```1.0```.

        use_language_weighted_sampler (bool):
            Enable / Disable the batch balancer by language. Defaults to ```False```.

        language_weighted_sampler_alpha (float):
            Number that control the influence of the language sampler weights. Defaults to ```1.0```.

        use_length_weighted_sampler (bool):
            Enable / Disable the batch balancer by audio length. If enabled the dataset will be divided
            into 10 buckets considering the min and max audio of the dataset. The sampler weights will be
            computed forcing to have the same quantity of data for each bucket in each training batch. Defaults to ```False```.

        length_weighted_sampler_alpha (float):
            Number that control the influence of the length sampler weights. Defaults to ```1.0```.
    """

    audio: BaseAudioConfig = field(default_factory=BaseAudioConfig)
    # training params
    batch_group_size: int = 0
    loss_masking: bool = None
    # dataloading
    min_audio_len: int = 1
    max_audio_len: int = float("inf")
    min_text_len: int = 1
    max_text_len: int = float("inf")
    compute_f0: bool = False
    compute_energy: bool = False
    compute_linear_spec: bool = False
    precompute_num_workers: int = 0
    use_noise_augment: bool = False
    start_by_longest: bool = False
    shuffle: bool = False
    drop_last: bool = False
    # dataset
    datasets: List[BaseDatasetConfig] = field(default_factory=lambda: [BaseDatasetConfig()])
    # optimizer
    optimizer: str = "radam"
    optimizer_params: dict = None
    # scheduler
    lr_scheduler: str = None
    lr_scheduler_params: dict = field(default_factory=lambda: {})
    # testing
    test_sentences: List[str] = field(default_factory=lambda: [])
    # evaluation
    eval_split_max_size: int = None
    eval_split_size: float = 0.01
    # weighted samplers
    use_speaker_weighted_sampler: bool = False
    speaker_weighted_sampler_alpha: float = 1.0
    use_language_weighted_sampler: bool = False
    language_weighted_sampler_alpha: float = 1.0
    use_length_weighted_sampler: bool = False
    length_weighted_sampler_alpha: float = 1.0
