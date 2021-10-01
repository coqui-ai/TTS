import json
import os
import random
import tarfile
from dataclasses import dataclass, field
from datetime import datetime
from typing import List
from coqpit import Coqpit
from multiprocessing import Pool

from TTS.stt.datasets import load_stt_samples
from TTS.config import BaseDatasetConfig


@dataclass
class ConvertedArgs(Coqpit):
    dataset_name: List[str] = field(
        default="",
        metadata={
            "help": "Name of the dataset(s) or the dataset format(s). Provided name(s) must be implemented in `stt.datasets.formatters`."
        },
    )
    dataset_path: List[str] = field(default_factory=list, metadata={"help": "Path(s) to the dataset(s)."})
    output_path: str = field(default="", metadata={"help": "Path to the output directory to save the tar shards."})
    num_shards: int = field(default=-1, metadata={"help": "Number of tarballs to create."})
    shuffle: bool = field(default=False, metadata={"help": "Shuffle the samples before tarring."})
    num_workers: int = field(default=1, metadata={"help": "Number of workers to use for parallelization."})


@dataclass
class TarMetadata(Coqpit):
    args: ConvertedArgs = field(default_factory=ConvertedArgs)
    created_date: str = field(default="", metadata={"help": "Date of creation of the tarball dataset."})
    num_samples_per_shard: int = field(default=0, metadata={"help": "Number of samples per tarball."})

    def __post_init__(self):
        self.created_date = self.get_date()

    @staticmethod
    def get_date():
        datetime.now().strftime("%m-%d-%Y %H-%M-%S")


def create_tar_shard(params):
    samples = params[0]
    output_path = params[1]
    shard_no = params[2]

    sharded_samples = []
    with tarfile.open(os.path.join(output_path, f'audio_{shard_no}.tar'), mode='w') as tar:
        count = {}
        for sample in samples:
            # We squash the filename since we do not preserve directory structure of audio files in the tarball.
            base, ext = os.path.splitext(sample['audio_file'])
            base = base.replace('/', '_')
            # Need the following replacement as long as WebDataset splits on first period
            base = base.replace('.', '_')
            squashed_filename = f'{base}{ext}'
            if squashed_filename not in count:
                tar.add(sample['audio_file'], arcname=squashed_filename)

            if "duration" in sample:
                duration = sample['duration']
            else:
                # TODO: not sure if this returns the right value
                duration = os.path.getsize(sample["audio_file"])
            count[squashed_filename] = 1
            sharded_sample = {
                'audio_file': squashed_filename,
                'duration': duration,
                'text': sample['text'],
                'shard_no': shard_no,  # Keep shard ID for recordkeeping
            }
            sharded_samples.append(sharded_sample)
    return sharded_samples


if __name__ == "__main__":
    # parse command line arguments
    args = ConvertedArgs()
    args.parse_args(arg_prefix="")
    os.makedirs(args.output_path, exist_ok=True)

    # create tarring metadata config
    metadata_config = TarMetadata(args=args)

    # create dataset configs
    dataset_configs = []
    for dataset_name, dataset_path in zip(args.dataset_name, args.dataset_path):
        dataset_config = BaseDatasetConfig(name=dataset_name, path=dataset_path)
        dataset_configs.append(dataset_config)

    # load dataset samples
    samples, _ = load_stt_samples(dataset_configs, eval_split=False)
    print(f" > Number of data samples: {len(samples)}")

    # shuffle samples
    if args.shuffle:
        print(" > Shuffling data samples...")
        random.shuffle(samples)

    # define shard sample indices
    start_indices = []
    end_indices = []
    shard_size = (len(samples) // args.num_shards)
    for i in range(args.num_shards):
        start_idx = shard_size * i
        end_idx = start_idx + shard_size
        print(f" > Shard {i}: {start_idx} --> {end_idx}")
        if end_idx > len(samples):
            # discard the last shard to keep shard size the same
            print(f"Have {len(samples) - end_idx} entries left over that will be discarded.")
        start_indices.append(start_idx)
        end_indices.append(end_idx)

    # create shards
    with Pool(args.num_workers) as pool:
        process_samples = [samples[start_idx:end_idx] for start_idx, end_idx in zip(start_indices, end_indices)]
        process_args = zip(process_samples, [args.output_path]*args.num_shards, range(args.num_shards))
        sharded_samples = pool.map(create_tar_shard, process_args)
    sharded_samples = [sample for sharded_sample in sharded_samples for sample in sharded_sample]
    print(f" > Total number of files sharded: {len(sharded_samples)}")

    # Write manifest
    metadata_path = os.path.join(args.output_path, 'coqui_tarred_dataset.json')
    with open(metadata_path, 'w', encoding="utf8") as m2:
        for entry in sharded_samples:
            json.dump(entry, m2)
            m2.write('\n')

    # Write metadata (default metadata for new datasets)
    metadata_config.num_samples_per_shard = shard_size
    metadata_path = os.path.join(args.output_path, 'metadata.json')
    metadata_config.save_json(metadata_path)