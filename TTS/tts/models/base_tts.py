import os
import random
from typing import Dict, List, Tuple, Union

import torch
import torch.distributed as dist
from coqpit import Coqpit
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from trainer.torch import DistributedSampler, DistributedSamplerWrapper

from TTS.model import BaseTrainerModel
from TTS.tts.datasets.dataset import TTSDataset
from TTS.tts.utils.data import get_length_balancer_weights
from TTS.tts.utils.languages import LanguageManager, get_language_balancer_weights
from TTS.tts.utils.speakers import SpeakerManager, get_speaker_balancer_weights, get_speaker_manager
from TTS.tts.utils.synthesis import synthesis
from TTS.tts.utils.visual import plot_alignment, plot_spectrogram

# pylint: skip-file


class BaseTTS(BaseTrainerModel):
    """Base `tts` class. Every new `tts` model must inherit this.

    It defines common `tts` specific functions on top of `Model` implementation.
    """

    MODEL_TYPE = "tts"

    def __init__(
        self,
        config: Coqpit,
        ap: "AudioProcessor",
        tokenizer: "TTSTokenizer",
        speaker_manager: SpeakerManager = None,
        language_manager: LanguageManager = None,
    ):
        super().__init__()
        self.config = config
        self.ap = ap
        self.tokenizer = tokenizer
        self.speaker_manager = speaker_manager
        self.language_manager = language_manager
        self._set_model_args(config)

    def _set_model_args(self, config: Coqpit):
        """Setup model args based on the config type (`ModelConfig` or `ModelArgs`).

        `ModelArgs` has all the fields reuqired to initialize the model architecture.

        `ModelConfig` has all the fields required for training, inference and containes `ModelArgs`.

        If the config is for training with a name like "*Config", then the model args are embeded in the
        config.model_args

        If the config is for the model with a name like "*Args", then we assign the directly.
        """
        # don't use isintance not to import recursively
        if "Config" in config.__class__.__name__:
            config_num_chars = (
                self.config.model_args.num_chars if hasattr(self.config, "model_args") else self.config.num_chars
            )
            num_chars = config_num_chars if self.tokenizer is None else self.tokenizer.characters.num_chars
            if "characters" in config:
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

    def get_aux_input_from_test_sentences(self, sentence_info):
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
        if self.speaker_manager is not None:
            if config.use_d_vector_file:
                if speaker_name is None:
                    d_vector = self.speaker_manager.get_random_embedding()
                else:
                    d_vector = self.speaker_manager.get_d_vector_by_name(speaker_name)
            elif config.use_speaker_embedding:
                if speaker_name is None:
                    speaker_id = self.speaker_manager.get_random_id()
                else:
                    speaker_id = self.speaker_manager.name_to_id[speaker_name]

        # get language id
        if self.language_manager is not None and config.use_language_embedding and language_name is not None:
            language_id = self.language_manager.name_to_id[language_name]

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
        text_input = batch["token_id"]
        text_lengths = batch["token_id_lengths"]
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
        energy = batch["energy"]
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
            "energy": energy,
            "language_ids": language_ids,
            "audio_unique_names": batch["audio_unique_names"],
        }

    def get_sampler(self, config: Coqpit, dataset: TTSDataset, num_gpus=1):
        weights = None
        data_items = dataset.samples

        if getattr(config, "use_language_weighted_sampler", False):
            alpha = getattr(config, "language_weighted_sampler_alpha", 1.0)
            print(" > Using Language weighted sampler with alpha:", alpha)
            weights = get_language_balancer_weights(data_items) * alpha

        if getattr(config, "use_speaker_weighted_sampler", False):
            alpha = getattr(config, "speaker_weighted_sampler_alpha", 1.0)
            print(" > Using Speaker weighted sampler with alpha:", alpha)
            if weights is not None:
                weights += get_speaker_balancer_weights(data_items) * alpha
            else:
                weights = get_speaker_balancer_weights(data_items) * alpha

        if getattr(config, "use_length_weighted_sampler", False):
            alpha = getattr(config, "length_weighted_sampler_alpha", 1.0)
            print(" > Using Length weighted sampler with alpha:", alpha)
            if weights is not None:
                weights += get_length_balancer_weights(data_items) * alpha
            else:
                weights = get_length_balancer_weights(data_items) * alpha

        if weights is not None:
            sampler = WeightedRandomSampler(weights, len(weights))
        else:
            sampler = None

        # sampler for DDP
        if sampler is None:
            sampler = DistributedSampler(dataset) if num_gpus > 1 else None
        else:  # If a sampler is already defined use this sampler and DDP sampler together
            sampler = DistributedSamplerWrapper(sampler) if num_gpus > 1 else sampler

        return sampler

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
        if is_eval and not config.run_eval:
            loader = None
        else:
            # setup multi-speaker attributes
            if self.speaker_manager is not None:
                if hasattr(config, "model_args"):
                    speaker_id_mapping = (
                        self.speaker_manager.name_to_id if config.model_args.use_speaker_embedding else None
                    )
                    d_vector_mapping = self.speaker_manager.embeddings if config.model_args.use_d_vector_file else None
                    config.use_d_vector_file = config.model_args.use_d_vector_file
                else:
                    speaker_id_mapping = self.speaker_manager.name_to_id if config.use_speaker_embedding else None
                    d_vector_mapping = self.speaker_manager.embeddings if config.use_d_vector_file else None
            else:
                speaker_id_mapping = None
                d_vector_mapping = None

            # setup multi-lingual attributes
            if self.language_manager is not None:
                language_id_mapping = self.language_manager.name_to_id if self.args.use_language_embedding else None
            else:
                language_id_mapping = None

            # init dataloader
            dataset = TTSDataset(
                outputs_per_step=config.r if "r" in config else 1,
                compute_linear_spec=config.model.lower() == "tacotron" or config.compute_linear_spec,
                compute_f0=config.get("compute_f0", False),
                f0_cache_path=config.get("f0_cache_path", None),
                compute_energy=config.get("compute_energy", False),
                energy_cache_path=config.get("energy_cache_path", None),
                samples=samples,
                ap=self.ap,
                return_wav=config.return_wav if "return_wav" in config else False,
                batch_group_size=0 if is_eval else config.batch_group_size * config.batch_size,
                min_text_len=config.min_text_len,
                max_text_len=config.max_text_len,
                min_audio_len=config.min_audio_len,
                max_audio_len=config.max_audio_len,
                phoneme_cache_path=config.phoneme_cache_path,
                precompute_num_workers=config.precompute_num_workers,
                use_noise_augment=False if is_eval else config.use_noise_augment,
                verbose=verbose,
                speaker_id_mapping=speaker_id_mapping,
                d_vector_mapping=d_vector_mapping if config.use_d_vector_file else None,
                tokenizer=self.tokenizer,
                start_by_longest=config.start_by_longest,
                language_id_mapping=language_id_mapping,
            )

            # wait all the DDP process to be ready
            if num_gpus > 1:
                dist.barrier()

            # sort input sequences from short to long
            dataset.preprocess_samples()

            # get samplers
            sampler = self.get_sampler(config, dataset, num_gpus)

            loader = DataLoader(
                dataset,
                batch_size=config.eval_batch_size if is_eval else config.batch_size,
                shuffle=config.shuffle if sampler is None else False,  # if there is no other sampler
                collate_fn=dataset.collate_fn,
                drop_last=config.drop_last,  # setting this False might cause issues in AMP training.
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
            d_vector = [self.speaker_manager.embeddings[name]["embedding"] for name in self.speaker_manager.embeddings]
            d_vector = (random.sample(sorted(d_vector), 1),)

        aux_inputs = {
            "speaker_id": None
            if not self.config.use_speaker_embedding
            else random.sample(sorted(self.speaker_manager.name_to_id.values()), 1),
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
        print(" | > Synthesizing test sentences.")
        test_audios = {}
        test_figures = {}
        test_sentences = self.config.test_sentences
        aux_inputs = self._get_test_aux_input()
        for idx, sen in enumerate(test_sentences):
            if isinstance(sen, list):
                aux_inputs = self.get_aux_input_from_test_sentences(sen)
                sen = aux_inputs["text"]
            outputs_dict = synthesis(
                self,
                sen,
                self.config,
                "cuda" in str(next(self.parameters()).device),
                speaker_id=aux_inputs["speaker_id"],
                d_vector=aux_inputs["d_vector"],
                style_wav=aux_inputs["style_wav"],
                use_griffin_lim=True,
                do_trim_silence=False,
            )
            test_audios["{}-audio".format(idx)] = outputs_dict["wav"]
            test_figures["{}-prediction".format(idx)] = plot_spectrogram(
                outputs_dict["outputs"]["model_outputs"], self.ap, output_fig=False
            )
            test_figures["{}-alignment".format(idx)] = plot_alignment(
                outputs_dict["outputs"]["alignments"], output_fig=False
            )
        return test_figures, test_audios

    def on_init_start(self, trainer):
        """Save the speaker.pth and language_ids.json at the beginning of the training. Also update both paths."""
        if self.speaker_manager is not None:
            output_path = os.path.join(trainer.output_path, "speakers.pth")
            self.speaker_manager.save_ids_to_file(output_path)
            trainer.config.speakers_file = output_path
            # some models don't have `model_args` set
            if hasattr(trainer.config, "model_args"):
                trainer.config.model_args.speakers_file = output_path
            trainer.config.save_json(os.path.join(trainer.output_path, "config.json"))
            print(f" > `speakers.pth` is saved to {output_path}.")
            print(" > `speakers_file` is updated in the config.json.")

        if self.language_manager is not None:
            output_path = os.path.join(trainer.output_path, "language_ids.json")
            self.language_manager.save_ids_to_file(output_path)
            trainer.config.language_ids_file = output_path
            if hasattr(trainer.config, "model_args"):
                trainer.config.model_args.language_ids_file = output_path
            trainer.config.save_json(os.path.join(trainer.output_path, "config.json"))
            print(f" > `language_ids.json` is saved to {output_path}.")
            print(" > `language_ids_file` is updated in the config.json.")
