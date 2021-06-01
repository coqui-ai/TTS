import json
import os
import random
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch

from TTS.config import load_config
from TTS.speaker_encoder.utils.generic_utils import setup_model
from TTS.utils.audio import AudioProcessor


def make_speakers_json_path(out_path):
    """Returns conventional speakers.json location."""
    return os.path.join(out_path, "speakers.json")


def load_speaker_mapping(out_path):
    """Loads speaker mapping if already present."""
    if os.path.splitext(out_path)[1] == ".json":
        json_file = out_path
    else:
        json_file = make_speakers_json_path(out_path)
    with open(json_file) as f:
        return json.load(f)


def save_speaker_mapping(out_path, speaker_mapping):
    """Saves speaker mapping if not yet present."""
    if out_path is not None:
        speakers_json_path = make_speakers_json_path(out_path)
        with open(speakers_json_path, "w") as f:
            json.dump(speaker_mapping, f, indent=4)


def get_speaker_manager(c, args, meta_data_train):
    """Inititalize and return a `SpeakerManager` based on config values"""
    speaker_manager = SpeakerManager()
    if c.use_speaker_embedding:
        speaker_manager.set_speaker_ids_from_data(meta_data_train)
        if args.restore_path:
            # restoring speaker manager from a previous run.
            if c.use_external_speaker_embedding_file:
                # restore speaker manager with the embedding file
                speakers_file = os.path.dirname(args.restore_path)
                if not os.path.exists(speakers_file):
                    print(
                        "WARNING: speakers.json was not found in restore_path, trying to use CONFIG.external_speaker_embedding_file"
                    )
                    if not os.path.exists(c.external_speaker_embedding_file):
                        raise RuntimeError(
                            "You must copy the file speakers.json to restore_path, or set a valid file in CONFIG.external_speaker_embedding_file"
                        )
                    speaker_manager.load_x_vectors_file(c.external_speaker_embedding_file)
                speaker_manager.set_x_vectors_from_file(speakers_file)
            elif not c.use_external_speaker_embedding_file:  # restor speaker manager with speaker ID file.
                speakers_file = os.path.dirname(args.restore_path)
                speaker_ids_from_data = speaker_manager.speaker_ids
                speaker_manager.set_speaker_ids_from_file(speakers_file)
                assert all(
                    speaker in speaker_manager.speaker_ids for speaker in speaker_ids_from_data
                ), " [!] You cannot introduce new speakers to a pre-trained model."
        elif c.use_external_speaker_embedding_file and c.external_speaker_embedding_file:
            # new speaker manager with external speaker embeddings.
            speaker_manager.set_x_vectors_from_file(c.external_speaker_embedding_file)
        elif (
            c.use_external_speaker_embedding_file and not c.external_speaker_embedding_file
        ):  # new speaker manager with speaker IDs file.
            raise "use_external_speaker_embedding_file is True, so you need pass a external speaker embedding file, run GE2E-Speaker_Encoder-ExtractSpeakerEmbeddings-by-sample.ipynb or AngularPrototypical-Speaker_Encoder-ExtractSpeakerEmbeddings-by-sample.ipynb notebook in notebooks/ folder"
        print(
            " > Training with {} speakers: {}".format(
                speaker_manager.num_speakers, ", ".join(speaker_manager.speaker_ids)
            )
        )
    return speaker_manager


class SpeakerManager:
    """It manages the multi-speaker setup for 🐸TTS models. It loads the speaker files and parses the information
    in a way that you can query. There are 3 different scenarios considered.

    1. Models using speaker embedding layers. The metafile only includes a mapping of speaker names to ids.
    2. Models using external embedding vectors (x vectors). The metafile includes a dictionary in the following
    format.

    ```
    {
        'clip_name.wav':{
            'name': 'speakerA',
            'embedding'[<x_vector_values>]
        },
        ...
    }
    ```

    3. Computing x vectors at inference with the speaker encoder. It loads the speaker encoder model and
    computes x vectors for a given instance.

    >>> >>> # load audio processor and speaker encoder
    >>> ap = AudioProcessor(**config.audio)
    >>> manager = SpeakerManager(encoder_model_path=encoder_model_path, encoder_config_path=encoder_config_path)
    >>> # load a sample audio and compute embedding
    >>> waveform = ap.load_wav(sample_wav_path)
    >>> mel = ap.melspectrogram(waveform)
    >>> x_vector = manager.compute_x_vector(mel.T)

    Args:
        x_vectors_file_path (str, optional): Path to the metafile including x vectors. Defaults to "".
        speaker_id_file_path (str, optional): Path to the metafile that maps speaker names to ids used by
        TTS models. Defaults to "".
        encoder_model_path (str, optional): Path to the speaker encoder model file. Defaults to "".
        encoder_config_path (str, optional): Path to the spealer encoder config file. Defaults to "".
    """

    def __init__(
        self,
        data_items: List[List[Any]] = None,
        x_vectors_file_path: str = "",
        speaker_id_file_path: str = "",
        encoder_model_path: str = "",
        encoder_config_path: str = "",
    ):

        self.data_items = []
        self.x_vectors = {}
        self.speaker_ids = []
        self.clip_ids = []
        self.speaker_encoder = None
        self.speaker_encoder_ap = None

        if data_items:
            self.speaker_ids, _ = self.parse_speakers_from_data(self.data_items)

        if x_vectors_file_path:
            self.set_x_vectors_from_file(x_vectors_file_path)

        if speaker_id_file_path:
            self.set_speaker_ids_from_file(speaker_id_file_path)

        if encoder_model_path and encoder_config_path:
            self.init_speaker_encoder(encoder_model_path, encoder_config_path)

    @staticmethod
    def _load_json(json_file_path: str) -> Dict:
        with open(json_file_path) as f:
            return json.load(f)

    @staticmethod
    def _save_json(json_file_path: str, data: dict) -> None:
        with open(json_file_path, "w") as f:
            json.dump(data, f, indent=4)

    @property
    def num_speakers(self):
        return len(self.speaker_ids)

    @property
    def x_vector_dim(self):
        """Dimensionality of x_vectors. If x_vectors are not loaded, returns zero."""
        if self.x_vectors:
            return len(self.x_vectors[list(self.x_vectors.keys())[0]]["embedding"])
        return 0

    @staticmethod
    def parse_speakers_from_data(items: list) -> Tuple[Dict, int]:
        """Parse speaker IDs from data samples retured by `load_meta_data()`.

        Args:
            items (list): Data sampled returned by `load_meta_data()`.

        Returns:
            Tuple[Dict, int]: speaker IDs and number of speakers.
        """
        speakers = sorted({item[2] for item in items})
        speaker_ids = {name: i for i, name in enumerate(speakers)}
        num_speakers = len(speaker_ids)
        return speaker_ids, num_speakers

    def set_speaker_ids_from_data(self, items: List) -> None:
        """Set speaker IDs from data samples.

        Args:
            items (List): Data sampled returned by `load_meta_data()`.
        """
        self.speaker_ids, _ = self.parse_speakers_from_data(items)

    def set_speaker_ids_from_file(self, file_path: str) -> None:
        """Set speaker IDs from a file.

        Args:
            file_path (str): Path to the file.
        """
        self.speaker_ids = self._load_json(file_path)

    def save_speaker_ids_to_file(self, file_path: str) -> None:
        """Save speaker IDs to a json file.

        Args:
            file_path (str): Path to the output file.
        """
        self._save_json(file_path, self.speaker_ids)

    def save_x_vectors_to_file(self, file_path: str) -> None:
        """Save x_vectors to a json file.

        Args:
            file_path (str): Path to the output file.
        """
        self._save_json(file_path, self.x_vectors)

    def set_x_vectors_from_file(self, file_path: str) -> None:
        """Load x_vectors from a json file.

        Args:
            file_path (str): Path to the target json file.
        """
        self.x_vectors = self._load_json(file_path)
        self.speaker_ids = list(set(sorted(x["name"] for x in self.x_vectors.values())))
        self.clip_ids = list(set(sorted(clip_name for clip_name in self.x_vectors.keys())))

    def get_x_vector_by_clip(self, clip_idx: str) -> List:
        """Get x_vector by clip ID.

        Args:
            clip_idx (str): Target clip ID.

        Returns:
            List: x_vector as a list.
        """
        return self.x_vectors[clip_idx]["embedding"]

    def get_x_vectors_by_speaker(self, speaker_idx: str) -> List[List]:
        """Get all x_vectors of a speaker.

        Args:
            speaker_idx (str): Target speaker ID.

        Returns:
            List[List]: all the x_vectors of the given speaker.
        """
        return [x["embedding"] for x in self.x_vectors.values() if x["name"] == speaker_idx]

    def get_mean_x_vector(self, speaker_idx: str, num_samples: int = None, randomize: bool = False) -> np.ndarray:
        """Get mean x_vector of a speaker ID.

        Args:
            speaker_idx (str): Target speaker ID.
            num_samples (int, optional): Number of samples to be averaged. Defaults to None.
            randomize (bool, optional): Pick random `num_samples`of x_vectors. Defaults to False.

        Returns:
            np.ndarray: Mean x_vector.
        """
        x_vectors = self.get_x_vectors_by_speaker(speaker_idx)
        if num_samples is None:
            x_vectors = np.stack(x_vectors).mean(0)
        else:
            assert len(x_vectors) >= num_samples, f" [!] speaker {speaker_idx} has number of samples < {num_samples}"
            if randomize:
                x_vectors = np.stack(random.choices(x_vectors, k=num_samples)).mean(0)
            else:
                x_vectors = np.stack(x_vectors[:num_samples]).mean(0)
        return x_vectors

    def get_speakers(self) -> List:
        return self.speaker_ids

    def get_clips(self) -> List:
        return sorted(self.x_vectors.keys())

    def init_speaker_encoder(self, model_path: str, config_path: str) -> None:
        """Initialize a speaker encoder model.

        Args:
            model_path (str): Model file path.
            config_path (str): Model config file path.
        """
        self.speaker_encoder_config = load_config(config_path)
        self.speaker_encoder = setup_model(self.speaker_encoder_config)
        self.speaker_encoder.load_checkpoint(config_path, model_path, True)
        self.speaker_encoder_ap = AudioProcessor(**self.speaker_encoder_config.audio)
        # normalize the input audio level and trim silences
        self.speaker_encoder_ap.do_sound_norm = True
        self.speaker_encoder_ap.do_trim_silence = True

    def compute_x_vector_from_clip(self, wav_file: Union[str, list]) -> list:
        """Compute a x_vector from a given audio file.

        Args:
            wav_file (Union[str, list]): Target file path.

        Returns:
            list: Computed x_vector.
        """

        def _compute(wav_file: str):
            waveform = self.speaker_encoder_ap.load_wav(wav_file, sr=self.speaker_encoder_ap.sample_rate)
            spec = self.speaker_encoder_ap.melspectrogram(waveform)
            spec = torch.from_numpy(spec.T)
            spec = spec.unsqueeze(0)
            x_vector = self.speaker_encoder.compute_embedding(spec)
            return x_vector

        if isinstance(wav_file, list):
            # compute the mean x_vector
            x_vectors = None
            for wf in wav_file:
                x_vector = _compute(wf)
                if x_vectors is None:
                    x_vectors = x_vector
                else:
                    x_vectors += x_vector
            return (x_vectors / len(wav_file))[0].tolist()
        x_vector = _compute(wav_file)
        return x_vector[0].tolist()

    def compute_x_vector(self, feats: Union[torch.Tensor, np.ndarray]) -> List:
        """Compute x_vector from features.

        Args:
            feats (Union[torch.Tensor, np.ndarray]): Input features.

        Returns:
            List: computed x_vector.
        """
        if isinstance(feats, np.ndarray):
            feats = torch.from_numpy(feats)
        if feats.ndim == 2:
            feats = feats.unsqueeze(0)
        return self.speaker_encoder.compute_embedding(feats)

    def run_umap(self):
        # TODO: implement speaker encoder
        raise NotImplementedError

    def plot_embeddings(self):
        # TODO: implement speaker encoder
        raise NotImplementedError
