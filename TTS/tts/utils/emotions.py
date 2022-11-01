import json
import os
from typing import Any, List

import fsspec
import numpy as np
import torch
from coqpit import Coqpit

from TTS.config import get_from_config_or_model_args_with_default
from TTS.tts.utils.managers import EmbeddingManager


class EmotionManager(EmbeddingManager):
    """Manage the emotions for emotional TTS. Load a datafile and parse the information
    in a way that can be queried by emotion or clip.

    There are 3 different scenarios considered:

    1. Models using emotion embedding layers. The datafile only maps emotion names to ids used by the embedding layer.
    2. Models using embeddings. The datafile includes a dictionary in the following format.

    ::

        {
            'clip_name.wav':{
                'name': 'emotionA',
                'embedding'[<embedding_values>]
            },
            ...
        }


    3. Computing the embeddings by the emotion encoder. It loads the emotion encoder model and
    computes the embeddings for a given clip or emotion.

    Args:
        embeddings_file_path (str, optional): Path to the metafile including x vectors. Defaults to "".
        emotion_id_file_path (str, optional): Path to the metafile that maps emotion names to ids used by
        TTS models. Defaults to "".
        encoder_model_path (str, optional): Path to the emotion encoder model file. Defaults to "".
        encoder_config_path (str, optional): Path to the spealer encoder config file. Defaults to "".

    Examples:
        >>> # load audio processor and emotion encoder
        >>> ap = AudioProcessor(**config.audio)
        >>> manager = EmotionManager(encoder_model_path=encoder_model_path, encoder_config_path=encoder_config_path)
        >>> # load a sample audio and compute embedding
        >>> embedding = manager.compute_embedding_from_clip(sample_wav_path)
    """

    def __init__(
        self,
        data_items: List[List[Any]] = None,
        embeddings_file_path: str = "",
        emotion_id_file_path: str = "",
        encoder_model_path: str = "",
        encoder_config_path: str = "",
        use_cuda: bool = False,
    ):
        super().__init__(
            embedding_file_path=embeddings_file_path,
            id_file_path=emotion_id_file_path,
            encoder_model_path=encoder_model_path,
            encoder_config_path=encoder_config_path,
            use_cuda=use_cuda,
        )

        if data_items:
            self.set_ids_from_data(data_items, parse_key="emotion_name")

    @property
    def emotion_names(self):
        return list(self.name_to_idx.keys())

    @property
    def num_emotions(self):
        return len(self.name_to_idx)

    @staticmethod
    def parse_ids_from_data(items: List, parse_key: str) -> Any:
        """Parse IDs from data samples retured by `load_tts_samples()`.

        Args:
            items (list): Data sampled returned by `load_tts_samples()`.
            parse_key (str): The key to being used to parse the data.
        Returns:
            Tuple[Dict]: speaker IDs.
        """
        classes = sorted({item[parse_key] for item in items})
        ids = {name: i for i, name in enumerate(classes)}
        return ids

    def set_ids_from_data(self, items: List, parse_key: str) -> Any:
        """Set IDs from data samples.

        Args:
            items (List): Data sampled returned by `load_tts_samples()`.
        """
        self.ids = self.parse_ids_from_data(items, parse_key=parse_key)

    def get_emotions(self) -> List:
        return self.ids

    @staticmethod
    def init_from_config(config: "Coqpit") -> "EmotionManager":
        """Initialize a emotion manager from config

        Args:
            config (Coqpit): Config object.

        Returns:
            EmotionEncoder: Emotion encoder object.
        """
        emotion_manager = None
        if (
            get_from_config_or_model_args_with_default(config, "use_emotion_embedding", False)
            or get_from_config_or_model_args_with_default(config, "use_prosody_enc_emo_classifier", False)
            or get_from_config_or_model_args_with_default(config, "use_text_enc_emo_classifier", False)
        ):
            if get_from_config_or_model_args_with_default(config, "emotions_ids_file", None):
                emotion_manager = EmotionManager(
                    emotion_id_file_path=get_from_config_or_model_args_with_default(config, "emotions_ids_file", None)
                )
            elif get_from_config_or_model_args_with_default(config, "use_emotion_vector_file", None):
                emotion_manager = EmotionManager(
                    embeddings_file_path=get_from_config_or_model_args_with_default(
                        config, "use_emotion_vector_file", None
                    )
                )

        if (
            get_from_config_or_model_args_with_default(config, "use_emotion_vector_file", False)
            or get_from_config_or_model_args_with_default(config, "use_prosody_enc_emo_classifier", False)
            or get_from_config_or_model_args_with_default(config, "use_text_enc_emo_classifier", False)
        ):
            if get_from_config_or_model_args_with_default(config, "emotion_vector_file", None):
                emotion_manager = EmotionManager(
                    embeddings_file_path=get_from_config_or_model_args_with_default(config, "emotion_vector_file", None)
                )

        return emotion_manager


def _set_file_path(path):
    """Find the emotions.json under the given path or the above it.
    Intended to band aid the different paths returned in restored and continued training."""
    path_restore = os.path.join(os.path.dirname(path), "emotions.json")
    path_continue = os.path.join(path, "emotions.json")
    fs = fsspec.get_mapper(path).fs
    if fs.exists(path_restore):
        return path_restore
    if fs.exists(path_continue):
        return path_continue
    raise FileNotFoundError(f" [!] `emotions.json` not found in {path}")


def load_emotion_mapping(out_path):
    """Loads speaker mapping if already present."""
    if os.path.splitext(out_path)[1] == ".json":
        json_file = out_path
    else:
        json_file = _set_file_path(out_path)
    with fsspec.open(json_file, "r") as f:
        return json.load(f)


def save_emotion_mapping(out_path, emotion_mapping):
    """Saves emotion mapping if not yet present."""
    if out_path is not None:
        emotions_json_path = _set_file_path(out_path)
        with fsspec.open(emotions_json_path, "w") as f:
            json.dump(emotion_mapping, f, indent=4)


def get_emotion_manager(c: Coqpit, restore_path: str = None, out_path: str = None) -> EmotionManager:
    """Initiate a `EmotionManager` instance by the provided config.

    Args:
        c (Coqpit): Model configuration.
        restore_path (str): Path to a previous training folder.
        out_path (str, optional): Save the generated emotion IDs to a output path. Defaults to None.

    Returns:
        EmotionManager: initialized and ready to use instance.
    """
    emotion_manager = EmotionManager()
    if restore_path:
        emotions_ids_file = _set_file_path(restore_path)
        # restoring emotion manager from a previous run.
        if c.use_external_emotions_embeddings:
            # restore emotion manager with the embedding file
            if not os.path.exists(emotions_ids_file):
                print(
                    "WARNING: emotions.json was not found in restore_path, trying to use CONFIG.external_emotions_embs_file"
                )
                if not os.path.exists(c.external_emotions_embs_file):
                    raise RuntimeError(
                        "You must copy the file emotions.json to restore_path, or set a valid file in CONFIG.external_emotions_embs_file"
                    )
                emotion_manager.load_embeddings_from_file(c.external_emotions_embs_file)
            emotion_manager.load_embeddings_from_file(emotions_ids_file)
        elif not c.use_external_emotions_embeddings:  # restor emotion manager with emotion ID file.
            emotion_manager.load_ids_from_file(emotions_ids_file)

    elif c.use_external_emotions_embeddings and c.external_emotions_embs_file:
        # new emotion manager with external emotion embeddings.
        emotion_manager.load_embeddings_from_file(c.external_emotions_embs_file)
    elif c.use_external_emotions_embeddings and not c.external_emotions_embs_file:
        raise "use_external_emotions_embeddings is True, so you need pass a external emotion embedding file."
    elif c.use_emotion_embedding:
        if "emotions_ids_file" in c and c.emotions_ids_file:
            emotion_manager.load_ids_from_file(c.emotions_ids_file)
        else:  # enable get ids from eternal embedding files
            emotion_manager.load_embeddings_from_file(c.external_emotions_embs_file)

    if emotion_manager.num_emotions > 0:
        print(
            " > Emotion manager is loaded with {} emotions: {}".format(
                emotion_manager.num_emotions, ", ".join(emotion_manager.ids)
            )
        )

    # save file if path is defined
    if out_path:
        out_file_path = os.path.join(out_path, "emotions.json")
        print(f" > Saving `emotions.json` to {out_file_path}.")
        if c.use_external_emotions_embeddings and c.external_emotions_embs_file:
            emotion_manager.save_embeddings_to_file(out_file_path)
        else:
            emotion_manager.save_ids_to_file(out_file_path)
    return emotion_manager


def get_speech_style_balancer_weights(items: list):
    style_names = np.array([item["speech_style"] for item in items])
    unique_style_names = np.unique(style_names).tolist()
    style_ids = [unique_style_names.index(s) for s in style_names]
    style_count = np.array([len(np.where(style_names == s)[0]) for s in unique_style_names])
    weight_style = 1.0 / style_count
    # get weight for each sample
    dataset_samples_weight = np.array([weight_style[s] for s in style_ids])
    # normalize
    dataset_samples_weight = dataset_samples_weight / np.linalg.norm(dataset_samples_weight)
    return torch.from_numpy(dataset_samples_weight).float()
