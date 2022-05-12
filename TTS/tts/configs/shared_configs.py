from dataclasses import asdict, dataclass, field
from typing import Dict, List

from coqpit import Coqpit, check_argument

from TTS.config import BaseAudioConfig, BaseDatasetConfig, BaseTrainingConfig


@dataclass
class GSTConfig(Coqpit):
    """Defines the Global Style Token Module

    Args:
        gst_style_input_wav (str):
            Path to the wav file used to define the style of the output speech at inference. Defaults to None.

        gst_style_input_weights (dict):
            Defines the weights for each style token used at inference. Defaults to None.

        gst_embedding_dim (int):
            Defines the size of the GST embedding vector dimensions. Defaults to 256.

        gst_num_heads (int):
            Number of attention heads used by the multi-head attention. Defaults to 4.

        gst_num_style_tokens (int):
            Number of style token vectors. Defaults to 10.
    """

    gst_style_input_wav: str = None
    gst_style_input_weights: dict = None
    gst_embedding_dim: int = 256
    gst_use_speaker_embedding: bool = False
    gst_num_heads: int = 4
    gst_num_style_tokens: int = 10

    def check_values(
        self,
    ):
        """Check config fields"""
        c = asdict(self)
        super().check_values()
        check_argument("gst_style_input_weights", c, restricted=False)
        check_argument("gst_style_input_wav", c, restricted=False)
        check_argument("gst_embedding_dim", c, restricted=True, min_val=0, max_val=1000)
        check_argument("gst_use_speaker_embedding", c, restricted=False)
        check_argument("gst_num_heads", c, restricted=True, min_val=2, max_val=10)
        check_argument("gst_num_style_tokens", c, restricted=True, min_val=1, max_val=1000)


@dataclass
class CharactersConfig(Coqpit):
    """Defines arguments for the `BaseCharacters` or `BaseVocabulary` and their subclasses.

    Args:
        characters_class (str):
            Defines the class of the characters used. If None, we pick ```Phonemes``` or ```Graphemes``` based on
            the configuration. Defaults to None.

        vocab_dict (dict):
            Defines the vocabulary dictionary used to encode the characters. Defaults to None.

        pad (str):
            characters in place of empty padding. Defaults to None.

        eos (str):
            characters showing the end of a sentence. Defaults to None.

        bos (str):
            characters showing the beginning of a sentence. Defaults to None.

        blank (str):
            Optional character used between characters by some models for better prosody. Defaults to `_blank`.

        characters (str):
            character set used by the model. Characters not in this list are ignored when converting input text to
            a list of sequence IDs. Defaults to None.

        punctuations (str):
            characters considered as punctuation as parsing the input sentence. Defaults to None.

        phonemes (str):
            characters considered as parsing phonemes. This is only for backwards compat. Use `characters` for new
            models. Defaults to None.

        is_unique (bool):
            remove any duplicate characters in the character lists. It is a bandaid for compatibility with the old
            models trained with character lists with duplicates. Defaults to True.

        is_sorted (bool):
            Sort the characters in alphabetical order. Defaults to True.
    """

    characters_class: str = None

    # using BaseVocabulary
    vocab_dict: Dict = None

    # using on BaseCharacters
    pad: str = None
    eos: str = None
    bos: str = None
    blank: str = None
    characters: str = None
    punctuations: str = None
    phonemes: str = None
    is_unique: bool = True  # for backwards compatibility of models trained with char sets with duplicates
    is_sorted: bool = True


@dataclass
class BaseTTSConfig(BaseTrainingConfig):
    """Shared parameters among all the tts models.

    Args:

        audio (BaseAudioConfig):
            Audio processor config object instance.

        use_phonemes (bool):
            enable / disable phoneme use.

        phonemizer (str):
            Name of the phonemizer to use. If set None, the phonemizer will be selected by `phoneme_language`.
            Defaults to None.

        phoneme_language (str):
            Language code for the phonemizer. You can check the list of supported languages by running
            `python TTS/tts/utils/text/phonemizers/__init__.py`. Defaults to None.

        compute_input_seq_cache (bool):
            enable / disable precomputation of the phoneme sequences. At the expense of some delay at the beginning of
            the training, It allows faster data loader time and precise limitation with `max_seq_len` and
            `min_seq_len`.

        text_cleaner (str):
            Name of the text cleaner used for cleaning and formatting transcripts.

        enable_eos_bos_chars (bool):
            enable / disable the use of eos and bos characters.

        test_senteces_file (str):
            Path to a txt file that has sentences used at test time. The file must have a sentence per line.

        phoneme_cache_path (str):
            Path to the output folder caching the computed phonemes for each sample.

        characters (CharactersConfig):
            Instance of a CharactersConfig class.

        batch_group_size (int):
            Size of the batch groups used for bucketing. By default, the dataloader orders samples by the sequence
            length for a more efficient and stable training. If `batch_group_size > 1` then it performs bucketing to
            prevent using the same batches for each epoch.

        loss_masking (bool):
            enable / disable masking loss values against padded segments of samples in a batch.

        sort_by_audio_len (bool):
            If true, dataloder sorts the data by audio length else sorts by the input text length. Defaults to `False`.

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

        compute_linear_spec (bool):
            If True data loader computes and returns linear spectrograms alongside the other data.

        precompute_num_workers (int):
            Number of workers to precompute features. Defaults to 0.

        use_noise_augment (bool):
            Augment the input audio with random noise.

        start_by_longest (bool):
            If True, the data loader will start loading the longest batch first. It is useful for checking OOM issues.
            Defaults to False.

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
    # phoneme settings
    use_phonemes: bool = False
    phonemizer: str = None
    phoneme_language: str = None
    compute_input_seq_cache: bool = False
    text_cleaner: str = None
    enable_eos_bos_chars: bool = False
    test_sentences_file: str = ""
    phoneme_cache_path: str = None
    # vocabulary parameters
    characters: CharactersConfig = None
    add_blank: bool = False
    # training params
    batch_group_size: int = 0
    loss_masking: bool = None
    # dataloading
    sort_by_audio_len: bool = False
    min_audio_len: int = 1
    max_audio_len: int = float("inf")
    min_text_len: int = 1
    max_text_len: int = float("inf")
    compute_f0: bool = False
    compute_linear_spec: bool = False
    precompute_num_workers: int = 0
    use_noise_augment: bool = False
    start_by_longest: bool = False
    # dataset
    datasets: List[BaseDatasetConfig] = field(default_factory=lambda: [BaseDatasetConfig()])
    # optimizer
    optimizer: str = "radam"
    optimizer_params: dict = None
    # scheduler
    lr_scheduler: str = ""
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
