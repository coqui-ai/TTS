import os
import random
from typing import Dict, List, Tuple

import torch
import torch.distributed as dist
from coqpit import Coqpit
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from TTS.model import BaseModel
from TTS.tts.configs.shared_configs import CharactersConfig
from TTS.tts.datasets.dataset import TTSDataset
from TTS.tts.utils.languages import LanguageManager, get_language_weighted_sampler
from TTS.tts.utils.speakers import SpeakerManager, get_speaker_manager, get_speaker_weighted_sampler
from TTS.tts.utils.synthesis import synthesis
from TTS.tts.utils.text import make_symbols
from TTS.tts.utils.visual import plot_alignment, plot_spectrogram

# pylint: skip-file


class BaseTTS(BaseModel):
    """Base `tts` class. Every new `tts` model must inherit this.

    It defines common `tts` specific functions on top of `Model` implementation.

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
            if "characters" in config:
                _, self.config, num_chars = self.get_characters(config)
                self.config.num_chars = num_chars
                if hasattr(self.config, "model_args"):
                    config.model_args.num_chars = num_chars
                    self.args = self.config.model_args
            else:
                self.config = config
                self.args = config.model_args
        elif "Args" in config.__class__.__name__:
            self.args = config
        else:
            raise ValueError("config must be either a *Config or *Args")

    @staticmethod
    def get_characters(config: Coqpit) -> str:
        # TODO: implement CharacterProcessor
        if config.characters is not None:
            symbols, phonemes = make_symbols(**config.characters)
        else:
            from TTS.tts.utils.text.symbols import parse_symbols, phonemes, symbols

            config.characters = CharactersConfig(**parse_symbols())
        model_characters = phonemes if config.use_phonemes else symbols
        num_chars = len(model_characters) + getattr(config, "add_blank", False)
        return model_characters, config, num_chars

    def get_speaker_manager(config: Coqpit, restore_path: str, data: List, out_path: str = None) -> SpeakerManager:
        return get_speaker_manager(config, restore_path, data, out_path)

    def init_multispeaker(self, config: Coqpit, data: List = None):
        """Initialize a speaker embedding layer if needen and define expected embedding channel size for defining
        `in_channels` size of the connected layers.

        This implementation yields 3 possible outcomes:

        1. If `config.use_speaker_embedding` and `config.use_d_vector_file are False, do nothing.
        2. If `config.use_d_vector_file` is True, set expected embedding channel size to `config.d_vector_dim` or 512.
        3. If `config.use_speaker_embedding`, initialize a speaker embedding layer with channel size of
        `config.d_vector_dim` or 512.

        You can override this function for new models.

        Args:
            config (Coqpit): Model configuration.
        """
        # set number of speakers
        if self.speaker_manager is not None:
            self.num_speakers = self.speaker_manager.num_speakers
        elif hasattr(config, "num_speakers"):
            self.num_speakers = config.num_speakers

        # set ultimate speaker embedding size
        if config.use_speaker_embedding or config.use_d_vector_file:
            self.embedded_speaker_dim = (
                config.d_vector_dim if "d_vector_dim" in config and config.d_vector_dim is not None else 512
            )
        # init speaker embedding layer
        if config.use_speaker_embedding and not config.use_d_vector_file:
            print(" > Init speaker_embedding layer.")
            self.speaker_embedding = nn.Embedding(self.num_speakers, self.embedded_speaker_dim)
            self.speaker_embedding.weight.data.normal_(0, 0.3)

    def get_aux_input(self, **kwargs) -> Dict:
        """Prepare and return `aux_input` used by `forward()`"""
        return {"speaker_id": None, "style_wav": None, "d_vector": None, "language_id": None}

    def get_aux_input_from_test_setences(self, sentence_info):
        if hasattr(self.config, "model_args"):
            config = self.config.model_args
        else:
            config = self.config

        # extract speaker and language info
        text, speaker_name, style_wav, language_name = None, None, None, None

        if isinstance(sentence_info, list):
            if len(sentence_info) == 1:
                text = sentence_info[0]
            elif len(sentence_info) == 2:
                text, speaker_name = sentence_info
            elif len(sentence_info) == 3:
                text, speaker_name, style_wav = sentence_info
            elif len(sentence_info) == 4:
                text, speaker_name, style_wav, language_name = sentence_info
        else:
            text = sentence_info

        # get speaker  id/d_vector
        speaker_id, d_vector, language_id = None, None, None
        if hasattr(self, "speaker_manager"):
            if config.use_d_vector_file:
                if speaker_name is None:
                    d_vector = self.speaker_manager.get_random_d_vector()
                else:
                    d_vector = self.speaker_manager.get_d_vector_by_speaker(speaker_name)
            elif config.use_speaker_embedding:
                if speaker_name is None:
                    speaker_id = self.speaker_manager.get_random_speaker_id()
                else:
                    speaker_id = self.speaker_manager.speaker_ids[speaker_name]

        # get language id
        if hasattr(self, "language_manager") and config.use_language_embedding and language_name is not None:
            language_id = self.language_manager.language_id_mapping[language_name]

        return {
            "text": text,
            "speaker_id": speaker_id,
            "style_wav": style_wav,
            "d_vector": d_vector,
            "language_id": language_id,
        }

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
        language_ids = batch["language_ids"]
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
            "language_ids": language_ids,
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

            # setup multi-speaker attributes
            if hasattr(self, "speaker_manager") and self.speaker_manager is not None:
                if hasattr(config, "model_args"):
                    speaker_id_mapping = (
                        self.speaker_manager.speaker_ids if config.model_args.use_speaker_embedding else None
                    )
                    d_vector_mapping = self.speaker_manager.d_vectors if config.model_args.use_d_vector_file else None
                    config.use_d_vector_file = config.model_args.use_d_vector_file
                else:
                    speaker_id_mapping = self.speaker_manager.speaker_ids if config.use_speaker_embedding else None
                    d_vector_mapping = self.speaker_manager.d_vectors if config.use_d_vector_file else None
            else:
                speaker_id_mapping = None
                d_vector_mapping = None

            # setup custom symbols if needed
            custom_symbols = None
            if hasattr(self, "make_symbols"):
                custom_symbols = self.make_symbols(self.config)

            if hasattr(self, "language_manager"):
                language_id_mapping = (
                    self.language_manager.language_id_mapping if self.args.use_language_embedding else None
                )
            else:
                language_id_mapping = None

            # init dataloader
            dataset = TTSDataset(
                outputs_per_step=config.r if "r" in config else 1,
                text_cleaner=config.text_cleaner,
                compute_linear_spec=config.model.lower() == "tacotron" or config.compute_linear_spec,
                compute_f0=config.get("compute_f0", False),
                f0_cache_path=config.get("f0_cache_path", None),
                meta_data=data_items,
                ap=ap,
                characters=config.characters,
                custom_symbols=custom_symbols,
                add_blank=config["add_blank"],
                return_wav=config.return_wav if "return_wav" in config else False,
                batch_group_size=0 if is_eval else config.batch_group_size * config.batch_size,
                min_seq_len=config.min_seq_len,
                max_seq_len=config.max_seq_len,
                phoneme_cache_path=config.phoneme_cache_path,
                use_phonemes=config.use_phonemes,
                phoneme_language=config.phoneme_language,
                enable_eos_bos=config.enable_eos_bos_chars,
                use_noise_augment=False if is_eval else config.use_noise_augment,
                verbose=verbose,
                speaker_id_mapping=speaker_id_mapping,
                d_vector_mapping=d_vector_mapping,
                language_id_mapping=language_id_mapping,
            )

            # pre-compute phonemes
            if config.use_phonemes and config.compute_input_seq_cache and rank in [None, 0]:
                if hasattr(self, "eval_data_items") and is_eval:
                    dataset.items = self.eval_data_items
                elif hasattr(self, "train_data_items") and not is_eval:
                    dataset.items = self.train_data_items
                else:
                    # precompute phonemes for precise estimate of sequence lengths.
                    # otherwise `dataset.sort_items()` uses raw text lengths
                    dataset.compute_input_seq(config.num_loader_workers)

                    # TODO: find a more efficient solution
                    # cheap hack - store items in the model state to avoid recomputing when reinit the dataset
                    if is_eval:
                        self.eval_data_items = dataset.items
                    else:
                        self.train_data_items = dataset.items

            # halt DDP processes for the main process to finish computing the phoneme cache
            if num_gpus > 1:
                dist.barrier()

            # sort input sequences from short to long
            dataset.sort_and_filter_items(config.get("sort_by_audio_len", default=False))

            # compute pitch frames and write to files.
            if config.compute_f0 and rank in [None, 0]:
                if not os.path.exists(config.f0_cache_path):
                    dataset.pitch_extractor.compute_pitch(
                        ap, config.get("f0_cache_path", None), config.num_loader_workers
                    )

            # halt DDP processes for the main process to finish computing the F0 cache
            if num_gpus > 1:
                dist.barrier()

            # load pitch stats computed above by all the workers
            if config.compute_f0:
                dataset.pitch_extractor.load_pitch_stats(config.get("f0_cache_path", None))

            # sampler for DDP
            sampler = DistributedSampler(dataset) if num_gpus > 1 else None

            # Weighted samplers
            assert not (
                num_gpus > 1 and getattr(config, "use_language_weighted_sampler", False)
            ), "language_weighted_sampler is not supported with DistributedSampler"
            assert not (
                num_gpus > 1 and getattr(config, "use_speaker_weighted_sampler", False)
            ), "speaker_weighted_sampler is not supported with DistributedSampler"

            if sampler is None:
                if getattr(config, "use_language_weighted_sampler", False):
                    print(" > Using Language weighted sampler")
                    sampler = get_language_weighted_sampler(dataset.items)
                elif getattr(config, "use_speaker_weighted_sampler", False):
                    print(" > Using Language weighted sampler")
                    sampler = get_speaker_weighted_sampler(dataset.items)

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

    def _get_test_aux_input(
        self,
    ) -> Dict:

        d_vector = None
        if self.config.use_d_vector_file:
            d_vector = [self.speaker_manager.d_vectors[name]["embedding"] for name in self.speaker_manager.d_vectors]
            d_vector = (random.sample(sorted(d_vector), 1),)

        aux_inputs = {
            "speaker_id": None
            if not self.config.use_speaker_embedding
            else random.sample(sorted(self.speaker_manager.speaker_ids.values()), 1),
            "d_vector": d_vector,
            "style_wav": None,  # TODO: handle GST style input
        }
        return aux_inputs

    def test_run(self, assets: Dict) -> Tuple[Dict, Dict]:
        """Generic test run for `tts` models used by `Trainer`.

        You can override this for a different behaviour.

        Args:
            assets (dict): A dict of training assets. For `tts` models, it must include `{'audio_processor': ap}`.

        Returns:
            Tuple[Dict, Dict]: Test figures and audios to be projected to Tensorboard.
        """
        ap = assets["audio_processor"]
        print(" | > Synthesizing test sentences.")
        test_audios = {}
        test_figures = {}
        test_sentences = self.config.test_sentences
        aux_inputs = self._get_test_aux_input()
        for idx, sen in enumerate(test_sentences):
            outputs_dict = synthesis(
                self,
                sen,
                self.config,
                "cuda" in str(next(self.parameters()).device),
                ap,
                speaker_id=aux_inputs["speaker_id"],
                d_vector=aux_inputs["d_vector"],
                style_wav=aux_inputs["style_wav"],
                enable_eos_bos_chars=self.config.enable_eos_bos_chars,
                use_griffin_lim=True,
                do_trim_silence=False,
            )
            test_audios["{}-audio".format(idx)] = outputs_dict["wav"]
            test_figures["{}-prediction".format(idx)] = plot_spectrogram(
                outputs_dict["outputs"]["model_outputs"], ap, output_fig=False
            )
            test_figures["{}-alignment".format(idx)] = plot_alignment(
                outputs_dict["outputs"]["alignments"], output_fig=False
            )
        return test_figures, test_audios

    def on_init_start(self, trainer):
        """Save the speaker.json and language_ids.json at the beginning of the training. Also update both paths."""
        if self.speaker_manager is not None:
            output_path = os.path.join(trainer.output_path, "speakers.json")
            self.speaker_manager.save_speaker_ids_to_file(output_path)
            trainer.config.speakers_file = output_path
            # some models don't have `model_args` set
            if hasattr(trainer.config, "model_args"):
                trainer.config.model_args.speakers_file = output_path
            trainer.config.save_json(os.path.join(trainer.output_path, "config.json"))
            print(f" > `speakers.json` is saved to {output_path}.")
            print(" > `speakers_file` is updated in the config.json.")

        if hasattr(self, "language_manager") and self.language_manager is not None:
            output_path = os.path.join(trainer.output_path, "language_ids.json")
            self.language_manager.save_language_ids_to_file(output_path)
            trainer.config.language_ids_file = output_path
            if hasattr(trainer.config, "model_args"):
                trainer.config.model_args.language_ids_file = output_path
            trainer.config.save_json(os.path.join(trainer.output_path, "config.json"))
            print(f" > `language_ids.json` is saved to {output_path}.")
            print(" > `language_ids_file` is updated in the config.json.")
