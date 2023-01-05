import json
import os
from typing import Any, Dict, List, Union

import fsspec
import numpy as np
import torch
from coqpit import Coqpit

from TTS.config import get_from_config_or_model_args_with_default
from TTS.tts.utils.managers import EmbeddingManager


class SpeakerManager(EmbeddingManager):
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
        super().__init__(
            embedding_file_path=d_vectors_file_path,
            id_file_path=speaker_id_file_path,
            encoder_model_path=encoder_model_path,
            encoder_config_path=encoder_config_path,
            use_cuda=use_cuda,
        )

        if data_items:
            self.set_ids_from_data(data_items, parse_key="speaker_name")

    @property
    def num_speakers(self):
        return len(self.name_to_id)

    @property
    def speaker_names(self):
        return list(self.name_to_id.keys())

    def get_speakers(self) -> List:
        return self.name_to_id

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
            speaker_manager = SpeakerManager()
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
            speaker_manager.set_ids_from_data(data, parse_key="speaker_name")
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
                    speaker_manager.load_embeddings_from_file(c.d_vector_file)
                speaker_manager.load_embeddings_from_file(speakers_file)
            elif not c.use_d_vector_file:  # restor speaker manager with speaker ID file.
                speaker_ids_from_data = speaker_manager.name_to_id
                speaker_manager.load_ids_from_file(speakers_file)
                assert all(
                    speaker in speaker_manager.name_to_id for speaker in speaker_ids_from_data
                ), " [!] You cannot introduce new speakers to a pre-trained model."
        elif c.use_d_vector_file and c.d_vector_file:
            # new speaker manager with external speaker embeddings.
            speaker_manager.load_embeddings_from_file(c.d_vector_file)
        elif c.use_d_vector_file and not c.d_vector_file:
            raise "use_d_vector_file is True, so you need pass a external speaker embedding file."
        elif c.use_speaker_embedding and "speakers_file" in c and c.speakers_file:
            # new speaker manager with speaker IDs file.
            speaker_manager.load_ids_from_file(c.speakers_file)

        if speaker_manager.num_speakers > 0:
            print(
                " > Speaker manager is loaded with {} speakers: {}".format(
                    speaker_manager.num_speakers, ", ".join(speaker_manager.name_to_id)
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
