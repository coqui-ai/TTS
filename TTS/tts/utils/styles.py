import json
import os
import random
from typing import Any, Dict, List, Tuple, Union

import fsspec
import numpy as np
import torch
from coqpit import Coqpit
from torch.utils.data.sampler import WeightedRandomSampler

from TTS.config import load_config
from TTS.speaker_encoder.utils.generic_utils import setup_speaker_encoder_model
from TTS.utils.audio import AudioProcessor


class StyleManager:
    """Manage the speakers for multi-speaker 🐸TTS models. Load a datafile and parse the information
    in a way that can be queried by speaker or clip.

    There are 3 different scenarios considered:

    1. Models using speaker embedding layers. The datafile only maps speaker names to ids used by the embedding layer.
    2. Models using d-vectors. The datafile includes a dictionary in the following format.

    ::

        {
            'clip_name.wav':{
                'name': 'speakerA',
                'embedding'[<d_vector_values>]
            },
            ...
        }


    3. Computing the d-vectors by the speaker encoder. It loads the speaker encoder model and
    computes the d-vectors for a given clip or speaker.

    Args:
        d_vectors_file_path (str, optional): Path to the metafile including x vectors. Defaults to "".
        speaker_id_file_path (str, optional): Path to the metafile that maps speaker names to ids used by
        TTS models. Defaults to "".
        encoder_model_path (str, optional): Path to the speaker encoder model file. Defaults to "".
        encoder_config_path (str, optional): Path to the spealer encoder config file. Defaults to "".

    Examples:
        >>> # load audio processor and speaker encoder
        >>> ap = AudioProcessor(**config.audio)
        >>> manager = SpeakerManager(encoder_model_path=encoder_model_path, encoder_config_path=encoder_config_path)
        >>> # load a sample audio and compute embedding
        >>> waveform = ap.load_wav(sample_wav_path)
        >>> mel = ap.melspectrogram(waveform)
        >>> d_vector = manager.compute_d_vector(mel.T)
    """

    def __init__(
        self,
        data_items: List[List[Any]] = None,
        d_vectors_file_path: str = "",
        style_id_file_path: str = "",
        encoder_model_path: str = "",
        encoder_config_path: str = "",
        use_cuda: bool = False,
    ):

        self.d_vectors = {}
        self.style_ids = {}
        self.clip_ids = []
        self.speaker_encoder = None
        self.speaker_encoder_ap = None
        self.use_cuda = use_cuda

        if data_items:
            self.style_ids, _ = self.parse_styles_from_data(data_items)

        if d_vectors_file_path:
            self.set_d_vectors_from_file(d_vectors_file_path)

        if style_id_file_path:
            self.set_style_ids_from_file(style_id_file_path)

        if encoder_model_path and encoder_config_path:
            self.init_speaker_encoder(encoder_model_path, encoder_config_path)

    @staticmethod
    def _load_json(json_file_path: str) -> Dict:
        with fsspec.open(json_file_path, "r") as f:
            return json.load(f)

    @staticmethod
    def _save_json(json_file_path: str, data: dict) -> None:
        with fsspec.open(json_file_path, "w") as f:
            json.dump(data, f, indent=4)

    @property
    def num_styles(self):
        return len(self.style_ids)

    @property
    def style_names(self):
        return list(self.style_ids.keys())

    @property
    def d_vector_dim(self):
        """Dimensionality of d_vectors. If d_vectors are not loaded, returns zero."""
        if self.d_vectors:
            return len(self.d_vectors[list(self.d_vectors.keys())[0]]["embedding"])
        return 0

    @staticmethod
    def parse_styles_from_data(items: list) -> Tuple[Dict, int]:
        """Parse speaker IDs from data samples retured by `load_tts_samples()`.

        Args:
            items (list): Data sampled returned by `load_tts_samples()`.

        Returns:
            Tuple[Dict, int]: speaker IDs and number of speakers.
        """
        speakers = sorted({item[2] for item in items})
        speaker_ids = {name: i for i, name in enumerate(speakers)}
        num_speakers = len(speaker_ids)
        return speaker_ids, num_speakers

    def set_style_ids_from_data(self, items: List) -> None:
        """Set speaker IDs from data samples.

        Args:
            items (List): Data sampled returned by `load_tts_samples()`.
        """
        self.style_ids, _ = self.parse_styles_from_data(items)

    def set_style_ids_from_file(self, file_path: str) -> None:
        """Set speaker IDs from a file.

        Args:
            file_path (str): Path to the file.
        """
        self.style_ids = self._load_json(file_path)

    def save_style_ids_to_file(self, file_path: str) -> None:
        """Save speaker IDs to a json file.

        Args:
            file_path (str): Path to the output file.
        """
        self._save_json(file_path, self.style_ids)

    def save_d_vectors_to_file(self, file_path: str) -> None:
        """Save d_vectors to a json file.

        Args:
            file_path (str): Path to the output file.
        """
        self._save_json(file_path, self.d_vectors)

    def set_d_vectors_from_file(self, file_path: str) -> None:
        """Load d_vectors from a json file.

        Args:
            file_path (str): Path to the target json file.
        """
        self.d_vectors = self._load_json(file_path)

        speakers = sorted({x["name"] for x in self.d_vectors.values()})
        self.speaker_ids = {name: i for i, name in enumerate(speakers)}

        self.clip_ids = list(set(sorted(clip_name for clip_name in self.d_vectors.keys())))

    def get_d_vector_by_clip(self, clip_idx: str) -> List:
        """Get d_vector by clip ID.

        Args:
            clip_idx (str): Target clip ID.

        Returns:
            List: d_vector as a list.
        """
        return self.d_vectors[clip_idx]["embedding"]

    def get_d_vectors_by_speaker(self, speaker_idx: str) -> List[List]:
        """Get all d_vectors of a speaker.

        Args:
            speaker_idx (str): Target speaker ID.

        Returns:
            List[List]: all the d_vectors of the given speaker.
        """
        return [x["embedding"] for x in self.d_vectors.values() if x["name"] == speaker_idx]

    def get_mean_d_vector(self, speaker_idx: str, num_samples: int = None, randomize: bool = False) -> np.ndarray:
        """Get mean d_vector of a speaker ID.

        Args:
            speaker_idx (str): Target speaker ID.
            num_samples (int, optional): Number of samples to be averaged. Defaults to None.
            randomize (bool, optional): Pick random `num_samples` of d_vectors. Defaults to False.

        Returns:
            np.ndarray: Mean d_vector.
        """
        d_vectors = self.get_d_vectors_by_speaker(speaker_idx)
        if num_samples is None:
            d_vectors = np.stack(d_vectors).mean(0)
        else:
            assert len(d_vectors) >= num_samples, f" [!] speaker {speaker_idx} has number of samples < {num_samples}"
            if randomize:
                d_vectors = np.stack(random.choices(d_vectors, k=num_samples)).mean(0)
            else:
                d_vectors = np.stack(d_vectors[:num_samples]).mean(0)
        return d_vectors

    def get_random_style_id(self) -> Any:
        """Get a random d_vector.

        Args:

        Returns:
            np.ndarray: d_vector.
        """
        if self.style_ids:
            return self.style_ids[random.choices(list(self.style_ids.keys()))[0]]

        return None

    def get_random_d_vector(self) -> Any:
        """Get a random D  ID.

        Args:

        Returns:
            np.ndarray: d_vector.
        """
        if self.d_vectors:
            return self.d_vectors[random.choices(list(self.d_vectors.keys()))[0]]["embedding"]

        return None

    def get_styles(self) -> List:
        return self.style_ids

    def get_clips(self) -> List:
        return sorted(self.d_vectors.keys())

    def init_speaker_encoder(self, model_path: str, config_path: str) -> None:
        """Initialize a speaker encoder model.

        Args:
            model_path (str): Model file path.
            config_path (str): Model config file path.
        """
        self.speaker_encoder_config = load_config(config_path)
        self.speaker_encoder = setup_speaker_encoder_model(self.speaker_encoder_config)
        self.speaker_encoder.load_checkpoint(config_path, model_path, eval=True, use_cuda=self.use_cuda)
        self.speaker_encoder_ap = AudioProcessor(**self.speaker_encoder_config.audio)

    def compute_d_vector_from_clip(self, wav_file: Union[str, List[str]]) -> list:
        """Compute a d_vector from a given audio file.

        Args:
            wav_file (Union[str, List[str]]): Target file path.

        Returns:
            list: Computed d_vector.
        """

        def _compute(wav_file: str):
            waveform = self.speaker_encoder_ap.load_wav(wav_file, sr=self.speaker_encoder_ap.sample_rate)
            if not self.speaker_encoder_config.model_params.get("use_torch_spec", False):
                m_input = self.speaker_encoder_ap.melspectrogram(waveform)
                m_input = torch.from_numpy(m_input)
            else:
                m_input = torch.from_numpy(waveform)

            if self.use_cuda:
                m_input = m_input.cuda()
            m_input = m_input.unsqueeze(0)
            d_vector = self.speaker_encoder.compute_embedding(m_input)
            return d_vector

        if isinstance(wav_file, list):
            # compute the mean d_vector
            d_vectors = None
            for wf in wav_file:
                d_vector = _compute(wf)
                if d_vectors is None:
                    d_vectors = d_vector
                else:
                    d_vectors += d_vector
            return (d_vectors / len(wav_file))[0].tolist()
        d_vector = _compute(wav_file)
        return d_vector[0].tolist()

    def compute_d_vector(self, feats: Union[torch.Tensor, np.ndarray]) -> List:
        """Compute d_vector from features.

        Args:
            feats (Union[torch.Tensor, np.ndarray]): Input features.

        Returns:
            List: computed d_vector.
        """
        if isinstance(feats, np.ndarray):
            feats = torch.from_numpy(feats)
        if feats.ndim == 2:
            feats = feats.unsqueeze(0)
        if self.use_cuda:
            feats = feats.cuda()
        return self.speaker_encoder.compute_embedding(feats)

    def run_umap(self):
        # TODO: implement speaker encoder
        raise NotImplementedError

    def plot_embeddings(self):
        # TODO: implement speaker encoder
        raise NotImplementedError


def _set_file_path(path):
    """Find the speakers.json under the given path or the above it.
    Intended to band aid the different paths returned in restored and continued training."""
    path_restore = os.path.join(os.path.dirname(path), "styles.json")
    path_continue = os.path.join(path, "styles.json")
    fs = fsspec.get_mapper(path).fs
    if fs.exists(path_restore):
        return path_restore
    if fs.exists(path_continue):
        return path_continue
    raise FileNotFoundError(f" [!] `styles.json` not found in {path}")


def load_style_mapping(out_path):
    """Loads speaker mapping if already present."""
    if os.path.splitext(out_path)[1] == ".json":
        json_file = out_path
    else:
        json_file = _set_file_path(out_path)
    with fsspec.open(json_file, "r") as f:
        return json.load(f)


def save_style_mapping(out_path, speaker_mapping):
    """Saves speaker mapping if not yet present."""
    if out_path is not None:
        speakers_json_path = _set_file_path(out_path)
        with fsspec.open(speakers_json_path, "w") as f:
            json.dump(speaker_mapping, f, indent=4)


def get_style_manager(c: Coqpit, data: List = None, restore_path: str = None, out_path: str = None) -> SpeakerManager:
    """Initiate a `SpeakerManager` instance by the provided config.

    Args:
        c (Coqpit): Model configuration.
        restore_path (str): Path to a previous training folder.
        data (List): Data samples used in training to infer speakers from. It must be provided if speaker embedding
            layers is used. Defaults to None.
        out_path (str, optional): Save the generated speaker IDs to a output path. Defaults to None.

    Returns:
        SpeakerManager: initialized and ready to use instance.
    """
    style_manager = StyleManager()
    if c.use_style_embedding:
        if data is not None:
            style_manager.set_style_ids_from_data(data)
        if restore_path:
            style_file = _set_file_path(restore_path)
            # restoring speaker manager from a previous run.
            if c.use_d_vector_file:
                # restore speaker manager with the embedding file
                if not os.path.exists(style_file):
                    print("WARNING: styles.json was not found in restore_path, trying to use CONFIG.d_vector_file")
                    if not os.path.exists(c.d_vector_file):
                        raise RuntimeError(
                            "You must copy the file styles.json to restore_path, or set a valid file in CONFIG.d_vector_file"
                        )
                    style_manager.load_d_vectors_file(c.d_vector_file)
                style_manager.set_d_vectors_from_file(style_file)
            elif not c.use_d_vector_file:  # restor speaker manager with speaker ID file.
                style_ids_from_data = style_manager.style_ids
                style_manager.set_speaker_ids_from_file(style_file)
                assert all(
                    style in style_manager.style_ids for style in style_ids_from_data
                ), " [!] You cannot introduce new styles to a pre-trained model."
        elif c.use_d_vector_file and c.d_vector_file:
            # new speaker manager with external speaker embeddings.
            style_manager.set_d_vectors_from_file(c.d_vector_file)
        elif c.use_d_vector_file and not c.d_vector_file:
            raise "use_d_vector_file is True, so you need pass a external speaker embedding file."
        elif c.use_style_embedding and "style_file" in c and c.style_file:
            # new speaker manager with speaker IDs file.
            style_manager.set_style_ids_from_file(c.style_file)

        if style_manager.num_styles > 0:
            print(
                " > Style manager is loaded with {} styles: {}".format(
                    style_manager.num_styles, ", ".join(style_manager.style_ids)
                )
            )

        # save file if path is defined
        if out_path:
            out_file_path = os.path.join(out_path, "speakers.json")
            print(f" > Saving `speakers.json` to {out_file_path}.")
            if c.use_d_vector_file and c.d_vector_file:
                speaker_manager.save_d_vectors_to_file(out_file_path)
            else:
                speaker_manager.save_speaker_ids_to_file(out_file_path)
    return speaker_manager


def get_speaker_weighted_sampler(items: list):
    speaker_names = np.array([item[2] for item in items])
    unique_speaker_names = np.unique(speaker_names).tolist()
    speaker_ids = [unique_speaker_names.index(l) for l in speaker_names]
    speaker_count = np.array([len(np.where(speaker_names == l)[0]) for l in unique_speaker_names])
    weight_speaker = 1.0 / speaker_count
    dataset_samples_weight = torch.from_numpy(np.array([weight_speaker[l] for l in speaker_ids])).double()
    return WeightedRandomSampler(dataset_samples_weight, len(dataset_samples_weight))
