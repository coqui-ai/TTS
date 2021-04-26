import json
import os
import random
from typing import Union

import numpy as np
import torch

from TTS.speaker_encoder.utils.generic_utils import setup_model
from TTS.utils.audio import AudioProcessor
from TTS.utils.io import load_config


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
    speakers_json_path = make_speakers_json_path(out_path)
    with open(speakers_json_path, "w") as f:
        json.dump(speaker_mapping, f, indent=4)


def get_speakers(items):
    """Returns a sorted, unique list of speakers in a given dataset."""
    speakers = {e[2] for e in items}
    return sorted(speakers)


def parse_speakers(c, args, meta_data_train, OUT_PATH):
    """ Returns number of speakers, speaker embedding shape and speaker mapping"""
    if c.use_speaker_embedding:
        speakers = get_speakers(meta_data_train)
        if args.restore_path:
            if c.use_external_speaker_embedding_file:  # if restore checkpoint and use External Embedding file
                prev_out_path = os.path.dirname(args.restore_path)
                speaker_mapping = load_speaker_mapping(prev_out_path)
                if not speaker_mapping:
                    print(
                        "WARNING: speakers.json was not found in restore_path, trying to use CONFIG.external_speaker_embedding_file"
                    )
                    speaker_mapping = load_speaker_mapping(c.external_speaker_embedding_file)
                    if not speaker_mapping:
                        raise RuntimeError(
                            "You must copy the file speakers.json to restore_path, or set a valid file in CONFIG.external_speaker_embedding_file"
                        )
                speaker_embedding_dim = len(speaker_mapping[list(speaker_mapping.keys())[0]]["embedding"])
            elif (
                not c.use_external_speaker_embedding_file
            ):  # if restore checkpoint and don't use External Embedding file
                prev_out_path = os.path.dirname(args.restore_path)
                speaker_mapping = load_speaker_mapping(prev_out_path)
                speaker_embedding_dim = None
                assert all(speaker in speaker_mapping for speaker in speakers), (
                    "As of now you, you cannot " "introduce new speakers to " "a previously trained model."
                )
        elif (
            c.use_external_speaker_embedding_file and c.external_speaker_embedding_file
        ):  # if start new train using External Embedding file
            speaker_mapping = load_speaker_mapping(c.external_speaker_embedding_file)
            speaker_embedding_dim = len(speaker_mapping[list(speaker_mapping.keys())[0]]["embedding"])
        elif (
            c.use_external_speaker_embedding_file and not c.external_speaker_embedding_file
        ):  # if start new train using External Embedding file and don't pass external embedding file
            raise "use_external_speaker_embedding_file is True, so you need pass a external speaker embedding file, run GE2E-Speaker_Encoder-ExtractSpeakerEmbeddings-by-sample.ipynb or AngularPrototypical-Speaker_Encoder-ExtractSpeakerEmbeddings-by-sample.ipynb notebook in notebooks/ folder"
        else:  # if start new train and don't use External Embedding file
            speaker_mapping = {name: i for i, name in enumerate(speakers)}
            speaker_embedding_dim = None
        save_speaker_mapping(OUT_PATH, speaker_mapping)
        num_speakers = len(speaker_mapping)
        print(" > Training with {} speakers: {}".format(len(speakers), ", ".join(speakers)))
    else:
        num_speakers = 0
        speaker_embedding_dim = None
        speaker_mapping = None

    return num_speakers, speaker_embedding_dim, speaker_mapping


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
        speaker_id_file_path (str, optional): Path to the metafile that maps speaker names to ids used by the
        TTS model. Defaults to "".
        encoder_model_path (str, optional): Path to the speaker encoder model file. Defaults to "".
        encoder_config_path (str, optional): Path to the spealer encoder config file. Defaults to "".
    """

    def __init__(
        self,
        x_vectors_file_path: str = "",
        speaker_id_file_path: str = "",
        encoder_model_path: str = "",
        encoder_config_path: str = "",
    ):

        self.x_vectors = None
        self.speaker_ids = None
        self.clip_ids = None
        self.speaker_encoder = None
        self.speaker_encoder_ap = None

        if x_vectors_file_path:
            self.load_x_vectors_file(x_vectors_file_path)

        if speaker_id_file_path:
            self.load_ids_file(speaker_id_file_path)

        if encoder_model_path and encoder_config_path:
            self.init_speaker_encoder(encoder_model_path, encoder_config_path)

    @staticmethod
    def _load_json(json_file_path: str):
        with open(json_file_path) as f:
            return json.load(f)

    @staticmethod
    def _save_json(json_file_path: str, data: dict):
        with open(json_file_path, "w") as f:
            json.dump(data, f, indent=4)

    @property
    def num_speakers(self):
        return len(self.speaker_ids)

    @property
    def x_vector_dim(self):
        return len(self.x_vectors[list(self.x_vectors.keys())[0]]["embedding"])

    def parser_speakers_from_items(self, items: list):
        speaker_ids = sorted({item[2] for item in items})
        self.speaker_ids = speaker_ids
        num_speakers = len(speaker_ids)
        return speaker_ids, num_speakers

    def save_ids_file(self, file_path: str):
        self._save_json(file_path, self.speaker_ids)

    def load_ids_file(self, file_path: str):
        self.speaker_ids = self._load_json(file_path)

    def save_x_vectors_file(self, file_path: str):
        self._save_json(file_path, self.x_vectors)

    def load_x_vectors_file(self, file_path: str):
        self.x_vectors = self._load_json(file_path)
        self.speaker_ids = list(set(sorted(x["name"] for x in self.x_vectors.values())))
        self.clip_ids = list(set(sorted(clip_name for clip_name in self.x_vectors.keys())))

    def get_x_vector_by_clip(self, clip_idx: str):
        return self.x_vectors[clip_idx]["embedding"]

    def get_x_vectors_by_speaker(self, speaker_idx: str):
        return [x["embedding"] for x in self.x_vectors.values() if x["name"] == speaker_idx]

    def get_mean_x_vector(self, speaker_idx: str, num_samples: int = None, randomize: bool = False):
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

    def get_speakers(self):
        return self.speaker_ids

    def get_clips(self):
        return sorted(self.x_vectors.keys())

    def init_speaker_encoder(self, model_path: str, config_path: str) -> None:
        self.speaker_encoder_config = load_config(config_path)
        self.speaker_encoder = setup_model(self.speaker_encoder_config)
        self.speaker_encoder.load_checkpoint(config_path, model_path, True)
        self.speaker_encoder_ap = AudioProcessor(**self.speaker_encoder_config.audio)
        # normalize the input audio level and trim silences
        self.speaker_encoder_ap.do_sound_norm = True
        self.speaker_encoder_ap.do_trim_silence = True

    def compute_x_vector_from_clip(self, wav_file: Union[str, list]) -> list:
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

    def compute_x_vector(self, feats):
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
