from typing import List

from coqpit import Coqpit
from torch.utils.data import Dataset

from TTS.utils.audio import AudioProcessor
from TTS.vocoder.datasets.gan_dataset import GANDataset
from TTS.vocoder.datasets.preprocess import load_wav_data, load_wav_feat_data
from TTS.vocoder.datasets.wavegrad_dataset import WaveGradDataset
from TTS.vocoder.datasets.wavernn_dataset import WaveRNNDataset


def setup_dataset(config: Coqpit, ap: AudioProcessor, is_eval: bool, data_items: List, verbose: bool) -> Dataset:
    if config.model.lower() in "gan":
        dataset = GANDataset(
            ap=ap,
            items=data_items,
            seq_len=config.seq_len,
            hop_len=ap.hop_length,
            pad_short=config.pad_short,
            conv_pad=config.conv_pad,
            return_pairs=config.diff_samples_for_G_and_D if "diff_samples_for_G_and_D" in config else False,
            is_training=not is_eval,
            return_segments=not is_eval,
            use_noise_augment=config.use_noise_augment,
            use_cache=config.use_cache,
            verbose=verbose,
        )
        dataset.shuffle_mapping()
    elif config.model.lower() == "wavegrad":
        dataset = WaveGradDataset(
            ap=ap,
            items=data_items,
            seq_len=config.seq_len,
            hop_len=ap.hop_length,
            pad_short=config.pad_short,
            conv_pad=config.conv_pad,
            is_training=not is_eval,
            return_segments=True,
            use_noise_augment=False,
            use_cache=config.use_cache,
            verbose=verbose,
        )
    elif config.model.lower() == "wavernn":
        dataset = WaveRNNDataset(
            ap=ap,
            items=data_items,
            seq_len=config.seq_len,
            hop_len=ap.hop_length,
            pad=config.model_params.pad,
            mode=config.model_params.mode,
            mulaw=config.model_params.mulaw,
            is_training=not is_eval,
            verbose=verbose,
        )
    else:
        raise ValueError(f" [!] Dataset for model {config.model.lower()} cannot be found.")
    return dataset
