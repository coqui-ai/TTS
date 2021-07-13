from typing import Dict, List, Tuple

import numpy as np
import torch
from coqpit import Coqpit
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from TTS.model import BaseModel
from TTS.tts.datasets import TTSDataset
from TTS.tts.utils.speakers import SpeakerManager, get_speaker_manager
from TTS.tts.utils.synthesis import synthesis
from TTS.tts.utils.text import make_symbols
from TTS.tts.utils.visual import plot_alignment, plot_spectrogram
from TTS.utils.audio import AudioProcessor

# pylint: skip-file


class BaseTTS(BaseModel):
    """Abstract `tts` class. Every new `tts` model must inherit this.

    It defines `tts` specific functions on top of `Model`.

    Notes on input/output tensor shapes:
        Any input or output tensor of the model must be shaped as

        - 3D tensors `batch x time x channels`
        - 2D tensors `batch x channels`
        - 1D tensors `batch x 1`
    """

    @staticmethod
    def get_characters(config: Coqpit) -> str:
        # TODO: implement CharacterProcessor
        if config.characters is not None:
            symbols, phonemes = make_symbols(**config.characters)
        else:
            from TTS.tts.utils.text.symbols import parse_symbols, phonemes, symbols

            config.characters = parse_symbols()
        model_characters = phonemes if config.use_phonemes else symbols
        num_chars = len(model_characters) + getattr(config, "add_blank", False)
        return model_characters, config, num_chars

    def get_speaker_manager(config: Coqpit, restore_path: str, data: List, out_path: str = None) -> SpeakerManager:
        return get_speaker_manager(config, restore_path, data, out_path)

    def init_multispeaker(self, config: Coqpit, data: List = None):
        """Initialize multi-speaker modules of a model. A model can be trained either with a speaker embedding layer
        or with external `d_vectors` computed from a speaker encoder model.

        If you need a different behaviour, override this function for your model.

        Args:
            config (Coqpit): Model configuration.
            data (List, optional): Dataset items to infer number of speakers. Defaults to None.
        """
        # init speaker manager
        self.speaker_manager = get_speaker_manager(config, data=data)
        self.num_speakers = self.speaker_manager.num_speakers
        # init speaker embedding layer
        if config.use_speaker_embedding and not config.use_d_vector_file:
            self.embedded_speaker_dim = (
                config.d_vector_dim if "d_vector_dim" in config and config.d_vector_dim is not None else 512
            )
            self.speaker_embedding = nn.Embedding(self.num_speakers, self.embedded_speaker_dim)
            self.speaker_embedding.weight.data.normal_(0, 0.3)

    def get_aux_input(self, **kwargs) -> Dict:
        """Prepare and return `aux_input` used by `forward()`"""
        pass

    def format_batch(self, batch: Dict) -> Dict:
        """Generic batch formatting for `TTSDataset`.

        You must override this if you use a custom dataset.

        Args:
            batch (Dict): [description]

        Returns:
            Dict: [description]
        """
        # setup input batch
        text_input = batch[0]
        text_lengths = batch[1]
        speaker_names = batch[2]
        linear_input = batch[3] if self.config.model.lower() in ["tacotron"] else None
        mel_input = batch[4]
        mel_lengths = batch[5]
        stop_targets = batch[6]
        item_idx = batch[7]
        d_vectors = batch[8]
        speaker_ids = batch[9]
        attn_mask = batch[10]
        max_text_length = torch.max(text_lengths.float())
        max_spec_length = torch.max(mel_lengths.float())

        # compute durations from attention masks
        durations = None
        if attn_mask is not None:
            durations = torch.zeros(attn_mask.shape[0], attn_mask.shape[2])
            for idx, am in enumerate(attn_mask):
                # compute raw durations
                c_idxs = am[:, : text_lengths[idx], : mel_lengths[idx]].max(1)[1]
                # c_idxs, counts = torch.unique_consecutive(c_idxs, return_counts=True)
                c_idxs, counts = torch.unique(c_idxs, return_counts=True)
                dur = torch.ones([text_lengths[idx]]).to(counts.dtype)
                dur[c_idxs] = counts
                # smooth the durations and set any 0 duration to 1
                # by cutting off from the largest duration indeces.
                extra_frames = dur.sum() - mel_lengths[idx]
                largest_idxs = torch.argsort(-dur)[:extra_frames]
                dur[largest_idxs] -= 1
                assert (
                    dur.sum() == mel_lengths[idx]
                ), f" [!] total duration {dur.sum()} vs spectrogram length {mel_lengths[idx]}"
                durations[idx, : text_lengths[idx]] = dur

        # set stop targets view, we predict a single stop token per iteration.
        stop_targets = stop_targets.view(text_input.shape[0], stop_targets.size(1) // self.config.r, -1)
        stop_targets = (stop_targets.sum(2) > 0.0).unsqueeze(2).float().squeeze(2)

        return {
            "text_input": text_input,
            "text_lengths": text_lengths,
            "speaker_names": speaker_names,
            "mel_input": mel_input,
            "mel_lengths": mel_lengths,
            "linear_input": linear_input,
            "stop_targets": stop_targets,
            "attn_mask": attn_mask,
            "durations": durations,
            "speaker_ids": speaker_ids,
            "d_vectors": d_vectors,
            "max_text_length": float(max_text_length),
            "max_spec_length": float(max_spec_length),
            "item_idx": item_idx,
        }

    def get_data_loader(
        self, config: Coqpit, ap: AudioProcessor, is_eval: bool, data_items: List, verbose: bool, num_gpus: int
    ) -> "DataLoader":
        if is_eval and not config.run_eval:
            loader = None
        else:
            # setup multi-speaker attributes
            if hasattr(self, "speaker_manager"):
                speaker_id_mapping = self.speaker_manager.speaker_ids if config.use_speaker_embedding else None
                d_vector_mapping = (
                    self.speaker_manager.d_vectors
                    if config.use_speaker_embedding and config.use_d_vector_file
                    else None
                )
            else:
                speaker_id_mapping = None
                d_vector_mapping = None

            # init dataloader
            dataset = TTSDataset(
                outputs_per_step=config.r if "r" in config else 1,
                text_cleaner=config.text_cleaner,
                compute_linear_spec=config.model.lower() == "tacotron",
                meta_data=data_items,
                ap=ap,
                characters=config.characters,
                add_blank=config["add_blank"],
                batch_group_size=0 if is_eval else config.batch_group_size * config.batch_size,
                min_seq_len=config.min_seq_len,
                max_seq_len=config.max_seq_len,
                phoneme_cache_path=config.phoneme_cache_path,
                use_phonemes=config.use_phonemes,
                phoneme_language=config.phoneme_language,
                enable_eos_bos=config.enable_eos_bos_chars,
                use_noise_augment=not is_eval,
                verbose=verbose,
                speaker_id_mapping=speaker_id_mapping,
                d_vector_mapping=d_vector_mapping
                if config.use_speaker_embedding and config.use_d_vector_file
                else None,
            )

            if config.use_phonemes and config.compute_input_seq_cache:
                # precompute phonemes to have a better estimate of sequence lengths.
                dataset.compute_input_seq(config.num_loader_workers)
            dataset.sort_items()

            sampler = DistributedSampler(dataset) if num_gpus > 1 else None
            loader = DataLoader(
                dataset,
                batch_size=config.eval_batch_size if is_eval else config.batch_size,
                shuffle=False,
                collate_fn=dataset.collate_fn,
                drop_last=False,
                sampler=sampler,
                num_workers=config.num_eval_loader_workers if is_eval else config.num_loader_workers,
                pin_memory=False,
            )
        return loader

    def test_run(self) -> Tuple[Dict, Dict]:
        """Generic test run for `tts` models used by `Trainer`.

        You can override this for a different behaviour.

        Returns:
            Tuple[Dict, Dict]: Test figures and audios to be projected to Tensorboard.
        """
        print(" | > Synthesizing test sentences.")
        test_audios = {}
        test_figures = {}
        test_sentences = self.config.test_sentences
        aux_inputs = self._get_aux_inputs()
        for idx, sen in enumerate(test_sentences):
            wav, alignment, model_outputs, _ = synthesis(
                self.model,
                sen,
                self.config,
                self.use_cuda,
                self.ap,
                speaker_id=aux_inputs["speaker_id"],
                d_vector=aux_inputs["d_vector"],
                style_wav=aux_inputs["style_wav"],
                enable_eos_bos_chars=self.config.enable_eos_bos_chars,
                use_griffin_lim=True,
                do_trim_silence=False,
            ).values()

            test_audios["{}-audio".format(idx)] = wav
            test_figures["{}-prediction".format(idx)] = plot_spectrogram(model_outputs, self.ap, output_fig=False)
            test_figures["{}-alignment".format(idx)] = plot_alignment(alignment, output_fig=False)
        return test_figures, test_audios
