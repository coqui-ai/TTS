import json
import random
from typing import Any, Dict, List, Tuple, Union

import fsspec
import numpy as np
import torch

from TTS.config import load_config
from TTS.encoder.utils.generic_utils import setup_encoder_model
from TTS.utils.audio import AudioProcessor


def load_file(path: str):
    if path.endswith(".json"):
        with fsspec.open(path, "r") as f:
            return json.load(f)
    elif path.endswith(".pth"):
        with fsspec.open(path, "rb") as f:
            return torch.load(f, map_location="cpu")
    else:
        raise ValueError("Unsupported file type")


def save_file(obj: Any, path: str):
    if path.endswith(".json"):
        with fsspec.open(path, "w") as f:
            json.dump(obj, f, indent=4)
    elif path.endswith(".pth"):
        with fsspec.open(path, "wb") as f:
            torch.save(obj, f)
    else:
        raise ValueError("Unsupported file type")


class BaseIDManager:
    """Base `ID` Manager class. Every new `ID` manager must inherit this.
    It defines common `ID` manager specific functions.
    """

    def __init__(self, id_file_path: str = ""):
        self.ids = {}

        if id_file_path:
            self.load_ids_from_file(id_file_path)

    @staticmethod
    def _load_json(json_file_path: str) -> Dict:
        with fsspec.open(json_file_path, "r") as f:
            return json.load(f)

    @staticmethod
    def _save_json(json_file_path: str, data: dict) -> None:
        with fsspec.open(json_file_path, "w") as f:
            json.dump(data, f, indent=4)

    def set_ids_from_data(self, items: List, parse_key: str) -> None:
        """Set IDs from data samples.

        Args:
            items (List): Data sampled returned by `load_tts_samples()`.
        """
        self.ids = self.parse_ids_from_data(items, parse_key=parse_key)

    def load_ids_from_file(self, file_path: str) -> None:
        """Set IDs from a file.

        Args:
            file_path (str): Path to the file.
        """
        self.ids = load_file(file_path)

    def save_ids_to_file(self, file_path: str) -> None:
        """Save IDs to a json file.

        Args:
            file_path (str): Path to the output file.
        """
        save_file(self.ids, file_path)

    def get_random_id(self) -> Any:
        """Get a random embedding.

        Args:

        Returns:
            np.ndarray: embedding.
        """
        if self.ids:
            return self.ids[random.choices(list(self.ids.keys()))[0]]

        return None

    @staticmethod
    def parse_ids_from_data(items: List, parse_key: str) -> Tuple[Dict]:
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


class EmbeddingManager(BaseIDManager):
    """Base `Embedding` Manager class. Every new `Embedding` manager must inherit this.
    It defines common `Embedding` manager specific functions.
    """

    def __init__(
        self,
        embedding_file_path: str = "",
        id_file_path: str = "",
        encoder_model_path: str = "",
        encoder_config_path: str = "",
        use_cuda: bool = False,
    ):
        super().__init__(id_file_path=id_file_path)

        self.embeddings = {}
        self.embeddings_by_names = {}
        self.clip_ids = []
        self.encoder = None
        self.encoder_ap = None
        self.use_cuda = use_cuda

        if embedding_file_path:
            self.load_embeddings_from_file(embedding_file_path)

        if encoder_model_path and encoder_config_path:
            self.init_encoder(encoder_model_path, encoder_config_path, use_cuda)

    @property
    def embedding_dim(self):
        """Dimensionality of embeddings. If embeddings are not loaded, returns zero."""
        if self.embeddings:
            return len(self.embeddings[list(self.embeddings.keys())[0]]["embedding"])
        return 0

    def save_embeddings_to_file(self, file_path: str) -> None:
        """Save embeddings to a json file.

        Args:
            file_path (str): Path to the output file.
        """
        save_file(self.embeddings, file_path)

    def load_embeddings_from_file(self, file_path: str) -> None:
        """Load embeddings from a json file.

        Args:
            file_path (str): Path to the target json file.
        """
        self.embeddings = load_file(file_path)

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

    def get_random_embedding(self) -> Any:
        """Get a random embedding.

        Args:

        Returns:
            np.ndarray: embedding.
        """
        if self.embeddings:
            return self.embeddings[random.choices(list(self.embeddings.keys()))[0]]["embedding"]

        return None

    def get_clips(self) -> List:
        return sorted(self.embeddings.keys())

    def init_encoder(self, model_path: str, config_path: str, use_cuda=False) -> None:
        """Initialize a speaker encoder model.

        Args:
            model_path (str): Model file path.
            config_path (str): Model config file path.
            use_cuda (bool, optional): Use CUDA. Defaults to False.
        """
        self.use_cuda = use_cuda
        self.encoder_config = load_config(config_path)
        self.encoder = setup_encoder_model(self.encoder_config)
        self.encoder_criterion = self.encoder.load_checkpoint(
            self.encoder_config, model_path, eval=True, use_cuda=use_cuda
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

    def compute_embeddings(self, feats: Union[torch.Tensor, np.ndarray]) -> List:
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
