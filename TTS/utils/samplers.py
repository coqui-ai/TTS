import torch
import numpy as np
from torch.utils.data.sampler import Sampler, WeightedRandomSampler, SubsetRandomSampler

def get_weighted_sampler(dataset, speaker_weighted_sampler, language_weighted_sampler):
    if speaker_weighted_sampler:
        # get speaker/language names
        speaker_names = np.array([item[2] for item in dataset.items])
        unique_speaker_names = np.unique(speaker_names).tolist()
        speaker_ids = [unique_speaker_names.index(s) for s in speaker_names]
        # count number samples by speaker/language
        speaker_count = np.array([len(np.where(speaker_names == s)[0]) for s in unique_speaker_names])
        # create weight
        weight_speaker = 1. / speaker_count
        samples_weight = np.array([weight_speaker[s] for s in speaker_ids])

    if language_weighted_sampler:
        language_names = np.array([item[3] for item in dataset.items])
        unique_language_names = np.unique(language_names).tolist()
        language_ids = [unique_language_names.index(l) for l in language_names]
        language_count = np.array([len(np.where(language_names == l)[0]) for l in unique_language_names])
        weight_language = 1. / language_count
        if speaker_weighted_sampler:
            samples_weight += np.array([weight_language[l] for l in language_ids])
        else:
            samples_weight = np.array([weight_language[l] for l in language_ids])

    dataset_samples_weight = torch.from_numpy(samples_weight).double()
    # create sampler
    return WeightedRandomSampler(dataset_samples_weight, len(dataset_samples_weight))

def get_perfect_language_sampler(dataset, c, is_val):
    assert not getattr(c, "gradual_training", False), 'batch size must be constant to use perfect sampler'
    return PerfectBatchSampler(dataset, getattr(c, "eval_batch_size")) if is_val else PerfectBatchSampler(dataset, getattr(c, "batch_size"))

class PerfectBatchSampler(Sampler):
    """Samples a mini-batch of indices for the grouped ConvolutionalEncoder.
    For L samples languages and batch size B produces a mini-batch with

    samples of a particular language L_i (random regardless speaker) 
    on the indices (into the mini-batch) i + k * L for k from 0 to B // L.
    
    Thus can be easily reshaped to a tensor of shape [B // L, L * C, ...]
    with groups consistent with languages.

    Arguments:
        dataset -- dataset to sample from
        batch_size -- total number of samples to be sampled in a mini-batch
    """

    def __init__(self, dataset, batch_size):

        languages = np.unique(np.array([item[3] for item in dataset.items])).tolist()

        assert batch_size % len(languages) == 0, (
            'Batch size must be divisible by number of languages.')

        label_indices = {}
        for idx in range(len(dataset)):
            label = dataset.items[idx][3]
            if label not in label_indices: 
                label_indices[label] = []
            label_indices[label].append(idx)

        self._samplers = [SubsetRandomSampler(label_indices[lang]) for lang in languages]

        self._batch_size = batch_size
        self.prepared_batch = []

    def __iter__(self):
        
        batch = []
        iters = [iter(s) for s in self._samplers]
        done = False
        
        while True:
            b = []
            for it in iters:
                idx = next(it, None)
                if idx is None:
                    done = True
                    break
                b.append(idx)
            if done: break
            batch += b
            if len(batch) == self._batch_size:
                yield batch
                batch = []
        
    def __len__(self):
        language_batch_size = self._batch_size // len(self._samplers)
        return min(((len(s) + language_batch_size - 1) // language_batch_size) for s in self._samplers)