import json
import os
import random
from typing import Any, Dict, List, Tuple, Union

import fsspec
import numpy as np
import torch
from coqpit import Coqpit

from TTS.config import get_from_config_or_model_args_with_default, load_config
from TTS.encoder.utils.generic_utils import setup_encoder_model
from TTS.utils.audio import AudioProcessor


class SpeakerManager:
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
        >>> d_vector = manager.compute_embeddings(mel.T)
    """

    def __init__(
        self,
        data_items: List[List[Any]] = None,
        d_vectors_file_path: str = "",
        speaker_id_file_path: str = "",
        encoder_model_path: str = "",
        encoder_config_path: str = "",
        use_cuda: bool = False,
    ):

        self.embeddings = {}
        self.ids = {}
        self.embeddings_by_names = {}
        self.clip_ids = []
        self.encoder = None
        self.encoder_ap = None
        self.use_cuda = use_cuda

        if data_items:
            self.ids, _ = self.parse_from_data(data_items)

        if d_vectors_file_path:
            self.set_embeddings_from_file(d_vectors_file_path)

        if speaker_id_file_path:
            self.set_ids_from_file(speaker_id_file_path)

        if encoder_model_path and encoder_config_path:
            self.init_encoder(encoder_model_path, encoder_config_path)

    @staticmethod
    def _load_json(json_file_path: str) -> Dict:
        with fsspec.open(json_file_path, "r") as f:
            return json.load(f)

    @staticmethod
    def _save_json(json_file_path: str, data: dict) -> None:
        with fsspec.open(json_file_path, "w") as f:
            json.dump(data, f, indent=4)

    @property
    def num_speakers(self):
        return len(self.ids)

    @property
    def speaker_names(self):
        return list(self.ids.keys())

    @property
    def embedding_dim(self):
        """Dimensionality of embeddings. If embeddings are not loaded, returns zero."""
        if self.embeddings:
            return len(self.embeddings[list(self.embeddings.keys())[0]]["embedding"])
        return 0

    @staticmethod
    def parse_from_data(items: list) -> Tuple[Dict, int]:
        """Parse speaker IDs from data samples retured by `load_tts_samples()`.

        Args:
            items (list): Data sampled returned by `load_tts_samples()`.

        Returns:
            Tuple[Dict, int]: speaker IDs and number of speakers.
        """
        speakers = sorted({item["speaker_name"] for item in items})
        speaker_ids = {name: i for i, name in enumerate(speakers)}
        num_speakers = len(speaker_ids)
        return speaker_ids, num_speakers

    def set_ids_from_data(self, items: List) -> None:
        """Set IDs from data samples.

        Args:
            items (List): Data sampled returned by `load_tts_samples()`.
        """
        self.ids, _ = self.parse_from_data(items)

    def set_ids_from_file(self, file_path: str) -> None:
        """Set speaker IDs from a file.

        Args:
            file_path (str): Path to the file.
        """
        self.ids = self._load_json(file_path)

    def save_ids_to_file(self, file_path: str) -> None:
        """Save speaker IDs to a json file.

        Args:
            file_path (str): Path to the output file.
        """
        self._save_json(file_path, self.ids)

    def save_embeddings_to_file(self, file_path: str) -> None:
        """Save embeddings to a json file.

        Args:
            file_path (str): Path to the output file.
        """
        self._save_json(file_path, self.embeddings)

    def set_embeddings_from_file(self, file_path: str) -> None:
        """Load embeddings from a json file.

        Args:
            file_path (str): Path to the target json file.
        """
        self.embeddings = self._load_json(file_path)

        speakers = sorted({x["name"] for x in self.embeddings.values()})
        self.ids = {name: i for i, name in enumerate(speakers)}

        self.clip_ids = list(set(sorted(clip_name for clip_name in self.embeddings.keys())))
        # cache embeddings_by_names for fast inference using a bigger speakers.json
        self.embeddings_by_names = self.get_embeddings_by_names()

    def get_embedding_by_clip(self, clip_idx: str) -> List:
        """Get embedding by clip ID.

        Args:
            clip_idx (str): Target clip ID.

        Returns:
            List: embedding as a list.
        """
        return self.embeddings[clip_idx]["embedding"]

    def get_embeddings_by_name(self, idx: str) -> List[List]:
        """Get all embeddings of a speaker.

        Args:
            idx (str): Target name.

        Returns:
            List[List]: all the embeddings of the given speaker.
        """
        return self.embeddings_by_names[idx]

    def get_embeddings_by_names(self) -> Dict:
        """Get all embeddings by names.

        Returns:
            Dict: all the embeddings of each speaker.
        """
        embeddings_by_names = {}
        for x in self.embeddings.values():
            if x["name"] not in embeddings_by_names.keys():
                embeddings_by_names[x["name"]] = [x["embedding"]]
            else:
                embeddings_by_names[x["name"]].append(x["embedding"])
        return embeddings_by_names

    def get_mean_embedding(self, idx: str, num_samples: int = None, randomize: bool = False) -> np.ndarray:
        """Get mean embedding of a idx.

        Args:
            idx (str): Target name.
            num_samples (int, optional): Number of samples to be averaged. Defaults to None.
            randomize (bool, optional): Pick random `num_samples` of embeddings. Defaults to False.

        Returns:
            np.ndarray: Mean embedding.
        """
        embeddings = self.get_embeddings_by_name(idx)
        if num_samples is None:
            embeddings = np.stack(embeddings).mean(0)
        else:
            assert len(embeddings) >= num_samples, f" [!] {idx} has number of samples < {num_samples}"
            if randomize:
                embeddings = np.stack(random.choices(embeddings, k=num_samples)).mean(0)
            else:
                embeddings = np.stack(embeddings[:num_samples]).mean(0)
        return embeddings

    def get_random_speaker_id(self) -> Any:
        """Get a random embedding.

        Args:

        Returns:
            np.ndarray: embedding.
        """
        if self.ids:
            return self.ids[random.choices(list(self.ids.keys()))[0]]

        return None

    def get_random_embedding(self) -> Any:
        """Get a random embedding.

        Args:

        Returns:
            np.ndarray: embedding.
        """
        if self.embeddings:
            return self.embeddings[random.choices(list(self.embeddings.keys()))[0]]["embedding"]

        return None

    def get_speakers(self) -> List:
        return self.ids

    def get_clips(self) -> List:
        return sorted(self.embeddings.keys())

    def init_encoder(self, model_path: str, config_path: str) -> None:
        """Initialize a speaker encoder model.

        Args:
            model_path (str): Model file path.
            config_path (str): Model config file path.
        """
        self.encoder_config = load_config(config_path)
        self.encoder = setup_encoder_model(self.encoder_config)
        self.speaker_encoder_criterion = self.speaker_encoder.load_checkpoint(
            self.encoder_config, model_path, eval=True, use_cuda=self.use_cuda
        )
        self.encoder_ap = AudioProcessor(**self.encoder_config.audio)

    def compute_embedding_from_clip(self, wav_file: Union[str, List[str]]) -> list:
        """Compute a embedding from a given audio file.

        Args:
            wav_file (Union[str, List[str]]): Target file path.

        Returns:
            list: Computed embedding.
        """

        def _compute(wav_file: str):
            waveform = self.encoder_ap.load_wav(wav_file, sr=self.encoder_ap.sample_rate)
            if not self.encoder_config.model_params.get("use_torch_spec", False):
                m_input = self.encoder_ap.melspectrogram(waveform)
                m_input = torch.from_numpy(m_input)
            else:
                m_input = torch.from_numpy(waveform)

            if self.use_cuda:
                m_input = m_input.cuda()
            m_input = m_input.unsqueeze(0)
            embedding = self.encoder.compute_embedding(m_input)
            return embedding

        if isinstance(wav_file, list):
            # compute the mean embedding
            embeddings = None
            for wf in wav_file:
                embedding = _compute(wf)
                if embeddings is None:
                    embeddings = embedding
                else:
                    embeddings += embedding
            return (embeddings / len(wav_file))[0].tolist()
        embedding = _compute(wav_file)
        return embedding[0].tolist()

    def compute_embedding(self, feats: Union[torch.Tensor, np.ndarray]) -> List:
        """Compute embedding from features.

        Args:
            feats (Union[torch.Tensor, np.ndarray]): Input features.

        Returns:
            List: computed embedding.
        """
        if isinstance(feats, np.ndarray):
            feats = torch.from_numpy(feats)
        if feats.ndim == 2:
            feats = feats.unsqueeze(0)
        if self.use_cuda:
            feats = feats.cuda()
        return self.encoder.compute_embedding(feats)

    def run_umap(self):
        # TODO: implement speaker encoder
        raise NotImplementedError

    def plot_embeddings(self):
        # TODO: implement speaker encoder
        raise NotImplementedError

    @staticmethod
    def init_from_config(config: "Coqpit", samples: Union[List[List], List[Dict]] = None) -> "SpeakerManager":
        """Initialize a speaker manager from config

        Args:
            config (Coqpit): Config object.
            samples (Union[List[List], List[Dict]], optional): List of data samples to parse out the speaker names.
                Defaults to None.

        Returns:
            SpeakerEncoder: Speaker encoder object.
        """
        speaker_manager = None
        if get_from_config_or_model_args_with_default(config, "use_speaker_embedding", False):
            if samples:
                speaker_manager = SpeakerManager(data_items=samples)
            if get_from_config_or_model_args_with_default(config, "speaker_file", None):
                speaker_manager = SpeakerManager(
                    speaker_id_file_path=get_from_config_or_model_args_with_default(config, "speaker_file", None)
                )
            if get_from_config_or_model_args_with_default(config, "speakers_file", None):
                speaker_manager = SpeakerManager(
                    speaker_id_file_path=get_from_config_or_model_args_with_default(config, "speakers_file", None)
                )

        if get_from_config_or_model_args_with_default(config, "use_d_vector_file", False):
            if get_from_config_or_model_args_with_default(config, "speakers_file", None):
                speaker_manager = SpeakerManager(
                    d_vectors_file_path=get_from_config_or_model_args_with_default(config, "speaker_file", None)
                )
            if get_from_config_or_model_args_with_default(config, "d_vector_file", None):
                speaker_manager = SpeakerManager(
                    d_vectors_file_path=get_from_config_or_model_args_with_default(config, "d_vector_file", None)
                )
        return speaker_manager


def _set_file_path(path):
    """Find the speakers.json under the given path or the above it.
    Intended to band aid the different paths returned in restored and continued training."""
    path_restore = os.path.join(os.path.dirname(path), "speakers.json")
    path_continue = os.path.join(path, "speakers.json")
    fs = fsspec.get_mapper(path).fs
    if fs.exists(path_restore):
        return path_restore
    if fs.exists(path_continue):
        return path_continue
    raise FileNotFoundError(f" [!] `speakers.json` not found in {path}")


def load_speaker_mapping(out_path):
    """Loads speaker mapping if already present."""
    if os.path.splitext(out_path)[1] == ".json":
        json_file = out_path
    else:
        json_file = _set_file_path(out_path)
    with fsspec.open(json_file, "r") as f:
        return json.load(f)


def save_speaker_mapping(out_path, speaker_mapping):
    """Saves speaker mapping if not yet present."""
    if out_path is not None:
        speakers_json_path = _set_file_path(out_path)
        with fsspec.open(speakers_json_path, "w") as f:
            json.dump(speaker_mapping, f, indent=4)


def get_speaker_manager(c: Coqpit, data: List = None, restore_path: str = None, out_path: str = None) -> SpeakerManager:
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
    speaker_manager = SpeakerManager()
    if c.use_speaker_embedding:
        if data is not None:
            speaker_manager.set_ids_from_data(data)
        if restore_path:
            speakers_file = _set_file_path(restore_path)
            # restoring speaker manager from a previous run.
            if c.use_d_vector_file:
                # restore speaker manager with the embedding file
                if not os.path.exists(speakers_file):
                    print("WARNING: speakers.json was not found in restore_path, trying to use CONFIG.d_vector_file")
                    if not os.path.exists(c.d_vector_file):
                        raise RuntimeError(
                            "You must copy the file speakers.json to restore_path, or set a valid file in CONFIG.d_vector_file"
                        )
                    speaker_manager.load_embeddings_file(c.d_vector_file)
                speaker_manager.set_embeddings_from_file(speakers_file)
            elif not c.use_d_vector_file:  # restor speaker manager with speaker ID file.
                speaker_ids_from_data = speaker_manager.ids
                speaker_manager.set_ids_from_file(speakers_file)
                assert all(
                    speaker in speaker_manager.ids for speaker in speaker_ids_from_data
                ), " [!] You cannot introduce new speakers to a pre-trained model."
        elif c.use_d_vector_file and c.d_vector_file:
            # new speaker manager with external speaker embeddings.
            speaker_manager.set_embeddings_from_file(c.d_vector_file)
        elif c.use_d_vector_file and not c.d_vector_file:
            raise "use_d_vector_file is True, so you need pass a external speaker embedding file."
        elif c.use_speaker_embedding and "speakers_file" in c and c.speakers_file:
            # new speaker manager with speaker IDs file.
            speaker_manager.set_ids_from_file(c.speakers_file)

        if speaker_manager.num_speakers > 0:
            print(
                " > Speaker manager is loaded with {} speakers: {}".format(
                    speaker_manager.num_speakers, ", ".join(speaker_manager.ids)
                )
            )

        # save file if path is defined
        if out_path:
            out_file_path = os.path.join(out_path, "speakers.json")
            print(f" > Saving `speakers.json` to {out_file_path}.")
            if c.use_d_vector_file and c.d_vector_file:
                speaker_manager.save_embeddings_to_file(out_file_path)
            else:
                speaker_manager.save_ids_to_file(out_file_path)
    return speaker_manager


def get_speaker_balancer_weights(items: list):
    speaker_names = np.array([item["speaker_name"] for item in items])
    unique_speaker_names = np.unique(speaker_names).tolist()
    speaker_ids = [unique_speaker_names.index(l) for l in speaker_names]
    speaker_count = np.array([len(np.where(speaker_names == l)[0]) for l in unique_speaker_names])
    weight_speaker = 1.0 / speaker_count
    dataset_samples_weight = np.array([weight_speaker[l] for l in speaker_ids])
    # normalize
    dataset_samples_weight = dataset_samples_weight / np.linalg.norm(dataset_samples_weight)
    return torch.from_numpy(dataset_samples_weight).float()
