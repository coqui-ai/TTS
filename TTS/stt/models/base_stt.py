import os
from typing import Dict, List, Tuple

import torch
import torch.distributed as dist
from coqpit import Coqpit
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from TTS.model import BaseModel
from TTS.stt.datasets.dataset import STTDataset
from TTS.stt.datasets.tokenizer import Tokenizer
from TTS.tts.datasets import TTSDataset
from TTS.tts.utils.speakers import SpeakerManager, get_speaker_manager
from TTS.tts.utils.synthesis import synthesis
from TTS.tts.utils.text import make_symbols
from TTS.tts.utils.visual import plot_alignment, plot_spectrogram
from TTS.utils.audio import AudioProcessor

# pylint: skip-file


class BaseSTT(BaseModel):
    """Abstract `stt` class. Every new `stt` model must inherit this.

    It defines `stt` specific functions on top of `Model`.

    Notes on input/output tensor shapes:
        Any input or output tensor of the model must be shaped as

        - 3D tensors `batch x time x channels`
        - 2D tensors `batch x channels`
        - 1D tensors `batch x 1`
    """

    def _set_model_args(self, config: Coqpit):
        """Setup model args based on the config type.

        If the config is for training with a name like "*Config", then the model args are embeded in the
        config.model_args

        If the config is for the model with a name like "*Args", then we assign the directly.
        """
        # don't use isintance not to import recursively
        if "Config" in config.__class__.__name__:
            if "vocabulary" in config and config.vocabulary is not None:
                # loading from DeepSpeechConfig
                self.config = config
                self.args = config.model_args
                self.args.n_tokens = len(self.config.vocabulary)
            else:
                # loading from DeepSpeechArgs
                self.config = config
                self.args = config.model_args
        else:
            self.args = config

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

    def get_aux_input(self, **kwargs) -> Dict:
        """Prepare and return `aux_input` used by `forward()`"""
        ...

    def format_batch(self, batch: Dict) -> Dict:
        """Generic batch formatting for `TTSDataset`.

        You must override this if you use a custom dataset.

        Args:
            batch (Dict): [description]

        Returns:
            Dict: [description]
        """
        # setup input batch
        text_input = batch["text"]
        text_lengths = batch["text_lengths"]
        speaker_names = batch["speaker_names"]
        linear_input = batch["linear"]
        mel_input = batch["mel"]
        mel_lengths = batch["mel_lengths"]
        stop_targets = batch["stop_targets"]
        item_idx = batch["item_idxs"]
        d_vectors = batch["d_vectors"]
        speaker_ids = batch["speaker_ids"]
        attn_mask = batch["attns"]
        waveform = batch["waveform"]
        pitch = batch["pitch"]
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

        # set stop targets wrt reduction factor
        stop_targets = stop_targets.view(text_input.shape[0], stop_targets.size(1) // self.config.r, -1)
        stop_targets = (stop_targets.sum(2) > 0.0).unsqueeze(2).float().squeeze(2)
        stop_target_lengths = torch.divide(mel_lengths, self.config.r).ceil_()

        return {
            "text_input": text_input,
            "text_lengths": text_lengths,
            "speaker_names": speaker_names,
            "mel_input": mel_input,
            "mel_lengths": mel_lengths,
            "linear_input": linear_input,
            "stop_targets": stop_targets,
            "stop_target_lengths": stop_target_lengths,
            "attn_mask": attn_mask,
            "durations": durations,
            "speaker_ids": speaker_ids,
            "d_vectors": d_vectors,
            "max_text_length": float(max_text_length),
            "max_spec_length": float(max_spec_length),
            "item_idx": item_idx,
            "waveform": waveform,
            "pitch": pitch,
        }

    def get_data_loader(
        self,
        config: Coqpit,
        assets: Dict,
        is_eval: bool,
        data_items: List,
        verbose: bool,
        num_gpus: int,
        rank: int = None,
    ) -> "DataLoader":
        if is_eval and not config.run_eval:
            loader = None
        else:
            ap = assets["audio_processor"]
            tokenizer = assets["tokenizer"]

            # init dataset
            dataset = STTDataset(
                samples=data_items,
                ap=ap,
                tokenizer=tokenizer,
                batch_group_size=config.batch_group_size,
                sort_by_audio_len=config.sort_by_audio_len,
                min_seq_len=config.min_seq_len,
                max_seq_len=config.max_seq_len,
                verbose=verbose,
                feature_extractor=config.feature_extractor,
            )

            # halt DDP processes for the main process to finish computing the phoneme cache
            if num_gpus > 1:
                dist.barrier()

            # sampler for DDP
            sampler = DistributedSampler(dataset, shuffle=True) if num_gpus > 1 else None

            # init dataloader
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

    # def test_run(self, ap) -> Tuple[Dict, Dict]:
    #     """Generic test run for `tts` models used by `Trainer`.

    #     You can override this for a different behaviour.

    #     Returns:
    #         Tuple[Dict, Dict]: Test figures and audios to be projected to Tensorboard.
    #     """
    #     print(" | > Synthesizing test sentences.")
    #     test_audios = {}
    #     test_figures = {}
    #     test_sentences = self.config.test_sentences
    #     aux_inputs = self.get_aux_input()
    #     for idx, sen in enumerate(test_sentences):
    #         outputs_dict = synthesis(
    #             self,
    #             sen,
    #             self.config,
    #             "cuda" in str(next(self.parameters()).device),
    #             ap,
    #             speaker_id=aux_inputs["speaker_id"],
    #             d_vector=aux_inputs["d_vector"],
    #             style_wav=aux_inputs["style_wav"],
    #             enable_eos_bos_chars=self.config.enable_eos_bos_chars,
    #             use_griffin_lim=True,
    #             do_trim_silence=False,
    #         )
    #         test_audios["{}-audio".format(idx)] = outputs_dict["wav"]
    #         test_figures["{}-prediction".format(idx)] = plot_spectrogram(
    #             outputs_dict["outputs"]["model_outputs"], ap, output_fig=False
    #         )
    #         test_figures["{}-alignment".format(idx)] = plot_alignment(
    #             outputs_dict["outputs"]["alignments"], output_fig=False
    #         )
    #     return test_figures, test_audios
