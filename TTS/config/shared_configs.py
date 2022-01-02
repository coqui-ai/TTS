from dataclasses import asdict, dataclass
from typing import List

from coqpit import Coqpit, check_argument


@dataclass
class BaseAudioConfig(Coqpit):
    """Base config to definge audio processing parameters. It is used to initialize
    ```TTS.utils.audio.AudioProcessor.```

    Args:
        fft_size (int):
            Number of STFT frequency levels aka.size of the linear spectogram frame. Defaults to 1024.

        win_length (int):
            Each frame of audio is windowed by window of length ```win_length``` and then padded with zeros to match
            ```fft_size```. Defaults to 1024.

        hop_length (int):
            Number of audio samples between adjacent STFT columns. Defaults to 1024.

        frame_shift_ms (int):
            Set ```hop_length``` based on milliseconds and sampling rate.

        frame_length_ms (int):
            Set ```win_length``` based on milliseconds and sampling rate.

        stft_pad_mode (str):
            Padding method used in STFT. 'reflect' or 'center'. Defaults to 'reflect'.

        sample_rate (int):
            Audio sampling rate. Defaults to 22050.

        resample (bool):
            Enable / Disable resampling audio to ```sample_rate```. Defaults to ```False```.

        preemphasis (float):
            Preemphasis coefficient. Defaults to 0.0.

        ref_level_db (int): 20
            Reference Db level to rebase the audio signal and ignore the level below. 20Db is assumed the sound of air.
            Defaults to 20.

        do_sound_norm (bool):
            Enable / Disable sound normalization to reconcile the volume differences among samples. Defaults to False.

        log_func (str):
            Numpy log function used for amplitude to DB conversion. Defaults to 'np.log10'.

        do_trim_silence (bool):
            Enable / Disable trimming silences at the beginning and the end of the audio clip. Defaults to ```True```.

        do_amp_to_db_linear (bool, optional):
            enable/disable amplitude to dB conversion of linear spectrograms. Defaults to True.

        do_amp_to_db_mel (bool, optional):
            enable/disable amplitude to dB conversion of mel spectrograms. Defaults to True.

        trim_db (int):
            Silence threshold used for silence trimming. Defaults to 45.

        do_rms_norm (bool, optional):
            enable/disable RMS volume normalization when loading an audio file. Defaults to False.

        db_level (int, optional):
            dB level used for rms normalization. The range is -99 to 0. Defaults to None.

        power (float):
            Exponent used for expanding spectrogra levels before running Griffin Lim. It helps to reduce the
            artifacts in the synthesized voice. Defaults to 1.5.

        griffin_lim_iters (int):
            Number of Griffing Lim iterations. Defaults to 60.

        num_mels (int):
            Number of mel-basis frames that defines the frame lengths of each mel-spectrogram frame. Defaults to 80.

        mel_fmin (float): Min frequency level used for the mel-basis filters. ~50 for male and ~95 for female voices.
            It needs to be adjusted for a dataset. Defaults to 0.

        mel_fmax (float):
            Max frequency level used for the mel-basis filters. It needs to be adjusted for a dataset.

        spec_gain (int):
            Gain applied when converting amplitude to DB. Defaults to 20.

        signal_norm (bool):
            enable/disable signal normalization. Defaults to True.

        min_level_db (int):
            minimum db threshold for the computed melspectrograms. Defaults to -100.

        symmetric_norm (bool):
            enable/disable symmetric normalization. If set True normalization is performed in the range [-k, k] else
            [0, k], Defaults to True.

        max_norm (float):
            ```k``` defining the normalization range. Defaults to 4.0.

        clip_norm (bool):
            enable/disable clipping the our of range values in the normalized audio signal. Defaults to True.

        stats_path (str):
            Path to the computed stats file. Defaults to None.
    """

    # stft parameters
    fft_size: int = 1024
    win_length: int = 1024
    hop_length: int = 256
    frame_shift_ms: int = None
    frame_length_ms: int = None
    stft_pad_mode: str = "reflect"
    # audio processing parameters
    sample_rate: int = 22050
    resample: bool = False
    preemphasis: float = 0.0
    ref_level_db: int = 20
    do_sound_norm: bool = False
    log_func: str = "np.log10"
    # silence trimming
    do_trim_silence: bool = True
    trim_db: int = 45
    # rms volume normalization
    do_rms_norm: bool = False
    db_level: float = None
    # griffin-lim params
    power: float = 1.5
    griffin_lim_iters: int = 60
    # mel-spec params
    num_mels: int = 80
    mel_fmin: float = 0.0
    mel_fmax: float = None
    spec_gain: int = 20
    do_amp_to_db_linear: bool = True
    do_amp_to_db_mel: bool = True
    # normalization params
    signal_norm: bool = True
    min_level_db: int = -100
    symmetric_norm: bool = True
    max_norm: float = 4.0
    clip_norm: bool = True
    stats_path: str = None

    def check_values(
        self,
    ):
        """Check config fields"""
        c = asdict(self)
        check_argument("num_mels", c, restricted=True, min_val=10, max_val=2056)
        check_argument("fft_size", c, restricted=True, min_val=128, max_val=4058)
        check_argument("sample_rate", c, restricted=True, min_val=512, max_val=100000)
        check_argument(
            "frame_length_ms",
            c,
            restricted=True,
            min_val=10,
            max_val=1000,
            alternative="win_length",
        )
        check_argument("frame_shift_ms", c, restricted=True, min_val=1, max_val=1000, alternative="hop_length")
        check_argument("preemphasis", c, restricted=True, min_val=0, max_val=1)
        check_argument("min_level_db", c, restricted=True, min_val=-1000, max_val=10)
        check_argument("ref_level_db", c, restricted=True, min_val=0, max_val=1000)
        check_argument("power", c, restricted=True, min_val=1, max_val=5)
        check_argument("griffin_lim_iters", c, restricted=True, min_val=10, max_val=1000)

        # normalization parameters
        check_argument("signal_norm", c, restricted=True)
        check_argument("symmetric_norm", c, restricted=True)
        check_argument("max_norm", c, restricted=True, min_val=0.1, max_val=1000)
        check_argument("clip_norm", c, restricted=True)
        check_argument("mel_fmin", c, restricted=True, min_val=0.0, max_val=1000)
        check_argument("mel_fmax", c, restricted=True, min_val=500.0, allow_none=True)
        check_argument("spec_gain", c, restricted=True, min_val=1, max_val=100)
        check_argument("do_trim_silence", c, restricted=True)
        check_argument("trim_db", c, restricted=True)


@dataclass
class BaseDatasetConfig(Coqpit):
    """Base config for TTS datasets.

    Args:
        name (str):
            Dataset name that defines the preprocessor in use. Defaults to None.

        path (str):
            Root path to the dataset files. Defaults to None.

        meta_file_train (str):
            Name of the dataset meta file. Or a list of speakers to be ignored at training for multi-speaker datasets.
            Defaults to None.

        ignored_speakers (List):
            List of speakers IDs that are not used at the training. Default None.

        language (str):
            Language code of the dataset. If defined, it overrides `phoneme_language`. Defaults to None.

        meta_file_val (str):
            Name of the dataset meta file that defines the instances used at validation.

        meta_file_attn_mask (str):
            Path to the file that lists the attention mask files used with models that require attention masks to
            train the duration predictor.
    """

    name: str = ""
    path: str = ""
    meta_file_train: str = ""
    ignored_speakers: List[str] = None
    language: str = ""
    meta_file_val: str = ""
    meta_file_attn_mask: str = ""

    def check_values(
        self,
    ):
        """Check config fields"""
        c = asdict(self)
        check_argument("name", c, restricted=True)
        check_argument("path", c, restricted=True)
        check_argument("meta_file_train", c, restricted=True)
        check_argument("meta_file_val", c, restricted=False)
        check_argument("meta_file_attn_mask", c, restricted=False)


@dataclass
class BaseTrainingConfig(Coqpit):
    """Base config to define the basic training parameters that are shared
    among all the models.

    Args:
        model (str):
            Name of the model that is used in the training.

        run_name (str):
            Name of the experiment. This prefixes the output folder name. Defaults to `coqui_tts`.

        run_description (str):
            Short description of the experiment.

        epochs (int):
            Number training epochs. Defaults to 10000.

        batch_size (int):
            Training batch size.

        eval_batch_size (int):
            Validation batch size.

        mixed_precision (bool):
            Enable / Disable mixed precision training. It reduces the VRAM use and allows larger batch sizes, however
            it may also cause numerical unstability in some cases.

        scheduler_after_epoch (bool):
            If true, run the scheduler step after each epoch else run it after each model step.

        run_eval (bool):
            Enable / Disable evaluation (validation) run. Defaults to True.

        test_delay_epochs (int):
            Number of epochs before starting to use evaluation runs. Initially, models do not generate meaningful
            results, hence waiting for a couple of epochs might save some time.

        print_eval (bool):
            Enable / Disable console logging for evalutaion steps. If disabled then it only shows the final values at
            the end of the evaluation. Default to ```False```.

        print_step (int):
            Number of steps required to print the next training log.

        log_dashboard (str): "tensorboard" or "wandb"
            Set the experiment tracking tool

        plot_step (int):
            Number of steps required to log training on Tensorboard.

        model_param_stats (bool):
            Enable / Disable logging internal model stats for model diagnostic. It might be useful for model debugging.
            Defaults to ```False```.

        project_name (str):
            Name of the project. Defaults to config.model

        wandb_entity (str):
            Name of W&B entity/team. Enables collaboration across a team or org.

        log_model_step (int):
            Number of steps required to log a checkpoint as W&B artifact

        save_step (int):ipt
            Number of steps required to save the next checkpoint.

        checkpoint (bool):
            Enable / Disable checkpointing.

        keep_all_best (bool):
            Enable / Disable keeping all the saved best models instead of overwriting the previous one. Defaults
            to ```False```.

        keep_after (int):
            Number of steps to wait before saving all the best models. In use if ```keep_all_best == True```. Defaults
            to 10000.

        num_loader_workers (int):
            Number of workers for training time dataloader.

        num_eval_loader_workers (int):
            Number of workers for evaluation time dataloader.

        output_path (str):
            Path for training output folder, either a local file path or other
            URLs supported by both fsspec and tensorboardX, e.g. GCS (gs://) or
            S3 (s3://) paths. The nonexist part of the given path is created
            automatically. All training artefacts are saved there.
    """

    model: str = None
    run_name: str = "coqui_tts"
    run_description: str = ""
    # training params
    epochs: int = 10000
    batch_size: int = None
    eval_batch_size: int = None
    mixed_precision: bool = False
    scheduler_after_epoch: bool = False
    # eval params
    run_eval: bool = True
    test_delay_epochs: int = 0
    print_eval: bool = False
    # logging
    dashboard_logger: str = "tensorboard"
    print_step: int = 25
    plot_step: int = 100
    model_param_stats: bool = False
    project_name: str = None
    log_model_step: int = None
    wandb_entity: str = None
    # checkpointing
    save_step: int = 10000
    checkpoint: bool = True
    keep_all_best: bool = False
    keep_after: int = 10000
    # dataloading
    num_loader_workers: int = 0
    num_eval_loader_workers: int = 0
    use_noise_augment: bool = False
    use_language_weighted_sampler: bool = False

    # paths
    output_path: str = None
    # distributed
    distributed_backend: str = "nccl"
    distributed_url: str = "tcp://localhost:54321"
