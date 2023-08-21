import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
import torchaudio
from coqpit import Coqpit
from encodec import EncodecModel
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from trainer.trainer_utils import get_optimizer, get_scheduler
from transformers import BertTokenizer

from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.datasets.dataset import _parse_sample
from TTS.tts.layers.bark.inference_funcs import (
    BarkHubertAudioTokenizer,
    codec_decode,
    convert_audio,
    generate_coarse,
    generate_fine,
    generate_text_semantic,
    generate_voice,
    load_voice,
)
from TTS.tts.layers.bark.load_model import load_model
from TTS.tts.layers.bark.model import GPT
from TTS.tts.layers.bark.model_fine import FineGPT
from TTS.tts.models.base_tts import BaseTTS


def load_audio(file_path, sr):
    """Load the audio file normalized in [-1, 1]

    Return Shapes:
        - x: :math:`[1, T]`
    """
    x, _sr = torchaudio.load(file_path)

    # resample if needed
    if sr != _sr:
        x = torchaudio.transforms.Resample(_sr, sr)(x)

    assert (x > 1).sum() + (x < -1).sum() == 0
    return x, sr


class BarkDataset(Dataset):
    def __init__(self, config, samples):
        super().__init__()
        self.samples = samples
        self.config = config

    def __getitem__(self, idx):
        item = self.samples[idx]
        raw_text = item["text"]

        wav, _ = load_audio(item["audio_file"], self.config.sample_rate)
        wav_filename = os.path.basename(item["audio_file"])

        return {
            "raw_text": raw_text,
            "text_len": len(raw_text),
            "wav": wav,
            "wav_len": wav.shape[1],
            "wav_file": wav_filename,
            "speaker_name": item["speaker_name"],
            "language_name": item["language"],
            "audio_unique_name": item["audio_unique_name"],
        }

    def __len__(self):
        return len(self.samples)

    @property
    def lengths(self):
        lens = []
        for item in self.samples:
            _, wav_file, *_ = _parse_sample(item)
            audio_len = os.path.getsize(wav_file) / 16 * 8  # assuming 16bit audio
            lens.append(audio_len)
        return lens

    def collate_fn(self, batch):
        """
        Return Shapes:
            - tokens: :math:`[B, T]`
            - token_lens :math:`[B]`
            - token_rel_lens :math:`[B]`
            - waveform: :math:`[B, 1, T]`
            - waveform_lens: :math:`[B]`
            - waveform_rel_lens: :math:`[B]`
            - speaker_names: :math:`[B]`
            - language_names: :math:`[B]`
            - audiofile_paths: :math:`[B]`
            - raw_texts: :math:`[B]`
            - audio_unique_names: :math:`[B]`
        """
        # convert list of dicts to dict of lists
        B = len(batch)
        batch = {k: [dic[k] for dic in batch] for k in batch[0]}

        _, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x.size(1) for x in batch["wav"]]), dim=0, descending=True
        )

        wav_lens = [w.shape[1] for w in batch["wav"]]
        wav_lens = torch.LongTensor(wav_lens)
        wav_lens_max = torch.max(wav_lens)
        wav_rel_lens = wav_lens / wav_lens_max

        wav_padded = torch.FloatTensor(B, 1, wav_lens_max)
        wav_padded = wav_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            wav = batch["wav"][i]
            wav_padded[i, :, : wav.size(1)] = torch.FloatTensor(wav)

        return {
            "waveform": wav_padded,  # (B x T)
            "waveform_lens": wav_lens,  # (B)
            "waveform_rel_lens": wav_rel_lens,
            "speaker_names": batch["speaker_name"],
            "language_names": batch["language_name"],
            "audio_files": batch["wav_file"],
            "raw_text": batch["raw_text"],
            "audio_unique_names": batch["audio_unique_name"],
        }


@dataclass
class BarkAudioConfig(Coqpit):
    sample_rate: int = 24000
    output_sample_rate: int = 24000


class Bark(BaseTTS):
    def __init__(
        self,
        config: Coqpit,
        tokenizer: BertTokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased"),
    ) -> None:
        super().__init__(config=config, ap=None, tokenizer=None, speaker_manager=None, language_manager=None)
        self.config.num_chars = len(tokenizer)
        self.tokenizer = tokenizer
        self.semantic_model = GPT(config.semantic_gpt_config)
        self.coarse_model = GPT(config.coarse_gpt_config)
        self.fine_model = FineGPT(config.fine_gpt_config)
        self.encodec = EncodecModel.encodec_model_24khz()
        self.encodec.set_target_bandwidth(6.0)
        self.semantic_tokenizer = BarkHubertAudioTokenizer(self.config, lazy_load=self.config.training_mode)

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def pad_token(self):
        if self.config.training_mode == "semantic":
            return self.config.SEMANTIC_PAD_TOKEN
        elif self.config.training_mode in ["coarse", "fine"]:
            return self.config.COARSE_SEMANTIC_PAD_TOKEN
        else:
            raise ValueError("Invalid training mode: {}".format(self.config.training_mode))

    def load_bark_models(self):
        self.semantic_model, self.config = load_model(
            ckpt_path=self.config.LOCAL_MODEL_PATHS["text"], device=self.device, config=self.config, model_type="text"
        )
        self.coarse_model, self.config = load_model(
            ckpt_path=self.config.LOCAL_MODEL_PATHS["coarse"],
            device=self.device,
            config=self.config,
            model_type="coarse",
        )
        self.fine_model, self.config = load_model(
            ckpt_path=self.config.LOCAL_MODEL_PATHS["fine"], device=self.device, config=self.config, model_type="fine"
        )

    def generate_coarse_fine_tokens(
        self,
        audio,
    ):
        if isinstance(audio, str):
            audio, sr = torchaudio.load(audio)
            audio = convert_audio(audio, sr, self.config.sample_rate, self.encodec.channels)
            audio = audio.unsqueeze(0).to(self.device)

        # Coarse and fine tokens
        with torch.no_grad():
            encoded_frames = self.encodec.encode(audio)
        codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1).squeeze()  # [n_q, T]
        return codes, codes[:2, :]  # fine, corse

    def generate_semantic_tokens(self, audio):
        return self.semantic_tokenizer.encode(audio, self.device)

    def text_to_semantic(
        self,
        text: str,
        history_prompt: Optional[str] = None,
        temp: float = 0.7,
        base=None,
        allow_early_stop=True,
        **kwargs,
    ):
        """Generate semantic array from text.

        Args:
            text: text to be turned into audio
            history_prompt: history choice for audio cloning
            temp: generation temperature (1.0 more diverse, 0.0 more conservative)

        Returns:
            numpy semantic array to be fed into `semantic_to_waveform`
        """
        x_semantic = generate_text_semantic(
            text,
            self,
            history_prompt=history_prompt,
            temp=temp,
            base=base,
            allow_early_stop=allow_early_stop,
            **kwargs,
        )
        return x_semantic

    def semantic_to_waveform(
        self,
        semantic_tokens: np.ndarray,
        history_prompt: Optional[str] = None,
        temp: float = 0.7,
        base=None,
    ):
        """Generate audio array from semantic input.

        Args:
            semantic_tokens: semantic token output from `text_to_semantic`
            history_prompt: history choice for audio cloning
            temp: generation temperature (1.0 more diverse, 0.0 more conservative)

        Returns:
            numpy audio array at sample frequency 24khz
        """
        x_coarse_gen = generate_coarse(
            semantic_tokens,
            self,
            history_prompt=history_prompt,
            temp=temp,
            base=base,
        )
        x_fine_gen = generate_fine(
            x_coarse_gen,
            self,
            history_prompt=history_prompt,
            temp=0.5,
            base=base,
        )
        audio_arr = codec_decode(x_fine_gen, self)
        return audio_arr, x_coarse_gen, x_fine_gen

    def generate_audio(
        self,
        text: str,
        history_prompt: Optional[str] = None,
        text_temp: float = 0.7,
        waveform_temp: float = 0.7,
        base=None,
        allow_early_stop=True,
        **kwargs,
    ):
        """Generate audio array from input text.

        Args:
            text: text to be turned into audio
            history_prompt: history choice for audio cloning
            text_temp: generation temperature (1.0 more diverse, 0.0 more conservative)
            waveform_temp: generation temperature (1.0 more diverse, 0.0 more conservative)

        Returns:
            numpy audio array at sample frequency 24khz
        """
        x_semantic = self.text_to_semantic(
            text,
            history_prompt=history_prompt,
            temp=text_temp,
            base=base,
            allow_early_stop=allow_early_stop,
            **kwargs,
        )
        audio_arr, c, f = self.semantic_to_waveform(
            x_semantic, history_prompt=history_prompt, temp=waveform_temp, base=base
        )
        return audio_arr, [x_semantic, c, f]

    def generate_voice(self, audio, speaker_id, voice_dir):
        """Generate a voice from the given audio and text.

        Args:
            audio (str): Path to the audio file.
            speaker_id (str): Speaker name.
            voice_dir (str): Path to the directory to save the generate voice.
        """
        if voice_dir is not None:
            voice_dirs = [voice_dir]
            try:
                _ = load_voice(speaker_id, voice_dirs)
            except (KeyError, FileNotFoundError):
                output_path = os.path.join(voice_dir, speaker_id + ".npz")
                os.makedirs(voice_dir, exist_ok=True)
                generate_voice(audio, self, output_path)

    def _set_voice_dirs(self, voice_dirs):
        def_voice_dir = None
        if isinstance(self.config.DEF_SPEAKER_DIR, str):
            os.makedirs(self.config.DEF_SPEAKER_DIR, exist_ok=True)
            if os.path.isdir(self.config.DEF_SPEAKER_DIR):
                def_voice_dir = self.config.DEF_SPEAKER_DIR
        _voice_dirs = [def_voice_dir] if def_voice_dir is not None else []
        if voice_dirs is not None:
            if isinstance(voice_dirs, str):
                voice_dirs = [voice_dirs]
            _voice_dirs = voice_dirs + _voice_dirs
        return _voice_dirs

    # TODO: remove config from synthesize
    def synthesize(
        self, text, config, speaker_id="random", voice_dirs=None, **kwargs
    ):  # pylint: disable=unused-argument
        """Synthesize speech with the given input text.

        Args:
            text (str): Input text.
            config (BarkConfig): Config with inference parameters.
            speaker_id (str): One of the available speaker names. If `random`, it generates a random speaker.
            speaker_wav (str): Path to the speaker audio file for cloning a new voice. It is cloned and saved in
                `voice_dirs` with the name `speaker_id`. Defaults to None.
            voice_dirs (List[str]): List of paths that host reference audio files for speakers. Defaults to None.
            **kwargs: Model specific inference settings used by `generate_audio()` and `TTS.tts.layers.bark.inference_funcs.generate_text_semantic().

        Returns:
            A dictionary of the output values with `wav` as output waveform, `deterministic_seed` as seed used at inference,
            `text_input` as text token IDs after tokenizer, `voice_samples` as samples used for cloning, `conditioning_latents`
            as latents used at inference.

        """
        speaker_id = "random" if speaker_id is None else speaker_id
        voice_dirs = self._set_voice_dirs(voice_dirs)
        history_prompt = load_voice(self, speaker_id, voice_dirs)
        outputs = self.generate_audio(text, history_prompt=history_prompt, **kwargs)
        return_dict = {
            "wav": outputs[0],
            "text_inputs": text,
        }

        return return_dict

    def format_batch(self, batch):
        """Tokenize input text.

        Args:
            batch (dict): batch of data to format

        Returns:
            formatted batch
        """
        tokenss = []
        max_len = 0
        for i, text in enumerate(batch["raw_text"]):
            tokens = np.array(self.tokenizer.encode(text, add_special_tokens=False)) + self.config.TEXT_ENCODING_OFFSET
            tokens = torch.from_numpy(tokens).long()
            tokenss.append(tokens)
            max_len = max(max_len, len(tokens))

        if self.config.training_mode == "semantic":
            # pad and collate into batch
            for i, tokens in enumerate(tokenss):
                tokenss[i] = torch.nn.functional.pad(tokens, (0, max_len - len(tokens)), value=self.pad_token)
            tokens = torch.stack(tokenss, dim=0)
            batch["input_ids"] = tokens[:, : self.config.train_semantic_data_settings["max_text_tokens_len"]]
        return batch

    def format_batch_on_device(self, batch):
        """Tokenize input audio.

        Args:
            batch (dict): Batch of input data.

        Returns:
            dict: Formatted batch.
        """
        # TODO: Make padding and truncation based on exact length of the waveforms
        if self.config.training_mode == "semantic":
            batch["semantic_tokens"] = self.generate_semantic_tokens(batch["waveform"][:, 0])[
                :, : self.config.max_semantic_tokens_len
            ]
        elif self.config.training_mode == "coarse":
            semantic_to_coarse_ratio = (
                self.config.COARSE_RATE_HZ / self.config.SEMANTIC_RATE_HZ * self.config.N_COARSE_CODEBOOKS
            )

            batch["semantic_tokens"] = self.generate_semantic_tokens(batch["waveform"][:, 0])[
                :, : self.config.train_coarse_data_settings["max_semantic_tokens_len"]
            ]
            batch["semantic_tokens"] = torch.nn.functional.pad(
                batch["semantic_tokens"], (0, 1), value=self.config.COARSE_INFER_TOKEN
            )

            batch["coarse_tokens"] = self.generate_coarse_fine_tokens(batch["waveform"])[1]
            batch["coarse_tokens"] = (
                batch["coarse_tokens"].flatten(start_dim=1)
                + self.config.CODEBOOK_SIZE
                + self.config.SEMANTIC_VOCAB_SIZE
            )
            batch["coarse_tokens"] = batch["coarse_tokens"][
                :, : self.config.train_coarse_data_settings["max_coarse_tokens_len"]
            ]
        elif self.config.training_mode == "fine":
            batch["coarse_tokens"], batch["fine_tokens"] = self.generate_coarse_fine_tokens(batch["waveform"])[
                :, : self.config.max_coarse_tokens_len
            ]
        return batch

    def train_step_semantic(self, batch: dict, criterion: torch.nn.Module) -> Tuple[Dict, Dict]:
        """Train semantic encoder"""
        tokens = batch["semantic_tokens"]
        target_tokens = tokens[:, 1:].contiguous()
        input_tokens = tokens[:, :-1].contiguous()

        inputs = torch.cat([batch["input_ids"], input_tokens], dim=1)
        logits = self.semantic_model(inputs)

        logits = logits[:, batch["input_ids"].size(1) :].contiguous()

        loss = criterion(logits.view(-1, self.config.semantic_gpt_config.output_vocab_size), target_tokens.view(-1))
        loss_dict = {"loss": loss}
        return {}, loss_dict

    def train_step_coarse(self, batch: dict, criterion: torch.nn.Module) -> Tuple[Dict, Dict]:
        """Train coarse encoder"""
        tokens = batch["coarse_tokens"]
        target_tokens = tokens[:, 1:].contiguous()
        input_tokens = tokens[:, :-1].contiguous()

        inputs = torch.cat([batch["semantic_tokens"], input_tokens], dim=1)
        logits = self.coarse_model(inputs)

        logits = logits[:, batch["semantic_tokens"].size(1) :].contiguous()

        loss = criterion(logits.view(-1, self.config.coarse_gpt_config.output_vocab_size), target_tokens.view(-1))
        loss_dict = {"loss": loss}
        return {}, loss_dict

    def train_step_fine(self):
        ...

    def train_step(self, *args, **kwargs):
        if self.config.training_mode == "semantic":
            return self.train_step_semantic(*args, **kwargs)
        elif self.config.training_mode == "coarse":
            return self.train_step_coarse(*args, **kwargs)
        elif self.config.training_mode == "fine":
            raise NotImplemented()

    def eval_step(self, *args, **kwargs):
        self.train_step(*args, **kwargs)

    def test_run(self, *args, **kwargs):
        ...

    def forward(self):
        ...

    def inference(self):
        ...

    def _get_test_aux_inputs(self):
        return None

    def get_criterion(self):
        return torch.nn.CrossEntropyLoss(ignore_index=self.pad_token)

    def get_optimizer(self):
        if self.config.training_mode == "semantic":
            optimizer = get_optimizer(
                self.config.optimizer, self.config.optimizer_params, self.config.lr, self.semantic_model
            )
        elif self.config.training_mode == "coarse":
            optimizer = get_optimizer(
                self.config.optimizer, self.config.optimizer_params, self.config.lr, self.coarse_model
            )
        elif self.config.training_mode == "fine":
            optimizer = get_optimizer(
                self.config.optimizer, self.config.optimizer_params, self.config.lr, self.fine_model
            )
        else:
            raise ValueError(" ❗ Invalid training mode: {}".format(self.config.training_mode))
        return optimizer

    def get_scheduler(self, optimizer):
        scheduler = get_scheduler(self.config.lr_scheduler, self.config.lr_scheduler_params, optimizer)
        return scheduler

    def get_data_loader(
        self,
        config: Coqpit,
        assets: Dict,
        is_eval: bool,
        samples: Union[List[Dict], List[List]],
        verbose: bool,
        num_gpus: int,
        rank: int = None,
    ) -> "DataLoader":
        from trainer.torch import DistributedSampler

        if is_eval and not config.run_eval:
            loader = None
        else:
            # init dataloader
            dataset = BarkDataset(
                config=self.config,
                samples=samples,
            )

            # wait all the DDP process to be ready
            if num_gpus > 1:
                dist.barrier()

            # init data loader
            sampler = DistributedSampler(dataset) if num_gpus > 1 else None
            loader = DataLoader(
                dataset,
                batch_size=config.eval_batch_size if is_eval else config.batch_size,
                shuffle=False,  # shuffle is done in the dataset.
                collate_fn=dataset.collate_fn,
                drop_last=True,  # setting this False might cause issues in AMP training.
                sampler=sampler,
                num_workers=config.num_eval_loader_workers if is_eval else config.num_loader_workers,
                pin_memory=True,
            )
        return loader

    @staticmethod
    def init_from_config(config: "BarkConfig", **kwargs):  # pylint: disable=unused-argument
        return Bark(config)

    # pylint: disable=unused-argument, redefined-builtin
    def load_checkpoint(
        self,
        config,
        checkpoint_dir,
        text_model_path=None,
        coarse_model_path=None,
        fine_model_path=None,
        eval=False,
        strict=True,
        **kwargs,
    ):
        """Load a model checkpoints from a directory. This model is with multiple checkpoint files and it
        expects to have all the files to be under the given `checkpoint_dir` with the rigth names.
        If eval is True, set the model to eval mode.

        Args:
            config (TortoiseConfig): The model config.
            checkpoint_dir (str): The directory where the checkpoints are stored.
            ar_checkpoint_path (str, optional): The path to the autoregressive checkpoint. Defaults to None.
            diff_checkpoint_path (str, optional): The path to the diffusion checkpoint. Defaults to None.
            clvp_checkpoint_path (str, optional): The path to the CLVP checkpoint. Defaults to None.
            vocoder_checkpoint_path (str, optional): The path to the vocoder checkpoint. Defaults to None.
            eval (bool, optional): Whether to set the model to eval mode. Defaults to False.
            strict (bool, optional): Whether to load the model strictly. Defaults to True.
        """
        text_model_path = text_model_path or os.path.join(checkpoint_dir, "text_2.pt")
        coarse_model_path = coarse_model_path or os.path.join(checkpoint_dir, "coarse_2.pt")
        fine_model_path = fine_model_path or os.path.join(checkpoint_dir, "fine_2.pt")

        self.config.LOCAL_MODEL_PATHS["text"] = text_model_path
        self.config.LOCAL_MODEL_PATHS["coarse"] = coarse_model_path
        self.config.LOCAL_MODEL_PATHS["fine"] = fine_model_path

        self.load_bark_models()

        if eval:
            self.eval()


if __name__ == "__main__":
    # from TTS.tts.configs.bark_config import BarkConfig

    # bark_config = BarkConfig()

    # bark_config.training_mode = "semantic"
    # bark_config.batch_size = 2

    # bark = Bark.init_from_config(bark_config)

    # # batch = {"waveform": torch.randn(2, 48000), "raw_text": ["hello world", "how are you"]}
    # # batch = bark.format_batch(batch)
    # # batch = bark.format_batch_on_device(batch)

    # from trainer import Trainer, TrainerArgs

    # dataset_config = BaseDatasetConfig(
    #     formatter="ljspeech", meta_file_train="metadata.csv", path="/data/TTS-public/tests/data/ljspeech/"
    # )

    # train_samples, eval_samples = load_tts_samples(
    #     dataset_config,
    #     eval_split=True,
    #     eval_split_max_size=4,
    #     eval_split_size=4,
    # )

    # trainer = Trainer(
    #     model=bark,
    #     config=bark_config,
    #     output_path="./",
    #     args=TrainerArgs(),
    #     train_samples=train_samples,
    #     eval_samples=eval_samples,
    # )
    # trainer.fit()

    from TTS.tts.configs.bark_config import BarkConfig

    bark_config = BarkConfig()

    bark_config.training_mode = "coarse"
    bark_config.batch_size = 2
    bark_config.run_eval = False
    bark_config.save_checkpoints = False
    bark_config.save_best_after = 100000
    bark_config.print_step = 1

    bark = Bark.init_from_config(bark_config)

    # batch = {"waveform": torch.randn(2, 48000), "raw_text": ["hello world", "how are you"]}
    # batch = bark.format_batch(batch)
    # batch = bark.format_batch_on_device(batch)

    from trainer import Trainer, TrainerArgs

    dataset_config = BaseDatasetConfig(
        formatter="ljspeech", meta_file_train="metadata.csv", path="/data/TTS-public/tests/data/ljspeech/"
    )

    train_samples, eval_samples = load_tts_samples(
        dataset_config,
        eval_split=True,
        eval_split_max_size=4,
        eval_split_size=4,
    )

    trainer = Trainer(
        model=bark,
        config=bark_config,
        output_path="./",
        args=TrainerArgs(),
        train_samples=train_samples,
        eval_samples=eval_samples,
    )
    trainer.fit()
