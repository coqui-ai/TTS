import os

import torch
from trainer import Trainer, TrainerArgs

from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import Vits, VitsArgs, VitsAudioConfig

torch.set_num_threads(24)

# pylint: disable=W0105
"""
    This recipe replicates the first experiment proposed in the YourTTS paper (https://arxiv.org/abs/2112.02418).
    YourTTS model is based on the VITS model however it uses external speaker embeddings extracted from a pre-trained speaker encoder and has small architecture changes.
    In addition, YourTTS can be trained in multilingual data, however, this recipe replicates the single language training using the VCTK dataset.
    The VitsArgs instance has commented parameters used to enable the multilingual training.
"""

# Name of the run for the Trainer
RUN_NAME = "YourTTS-EN-VCTK"

# Path where you want to save the models outputs (configs, checkpoints and tensorboard logs)
OUT_PATH = os.path.dirname(os.path.abspath(__file__))  # "/raid/coqui/Checkpoints/original-YourTTS/"

# If you want to do transfer learning and speedup your training you can set here the path to the original YourTTS model
RESTORE_PATH = None  # "/raid/coqui/Checkpoints/YourTTS/checkpoint.pth"

# This paramter is usefull to debug, it skips the training epochs and just do the evaluation  and produce the test sentences
SKIP_TRAIN_EPOCH = False

# Set here the batch size to be used in training and evaluation
BATCH_SIZE = 32

# To get the speakers.json or speakers.pth you need to follow the steps described at: https://github.com/Edresson/YourTTS#reproducibility
# or you can check the extract embedding script guidelines here: https://github.com/coqui-ai/TTS/blob/dev/TTS/bin/compute_embeddings.py#L20
D_VECTOR_FILES = [
    "/raid/datasets/VCTK/speakers.json",
]

# Change our dataset paths to the VCTK dataset or replace it for others
# init configs
vctk_config = BaseDatasetConfig(
    formatter="vctk", dataset_name="vctk", meta_file_train="metadata.csv", path="/raid/datasets/VCTK/", language="en"
)

# add here all datasets configs, in our case we just want to train with the VCTK dataset then we need to add just VCTK
datasets_list = [vctk_config]

# Audio config used in training. Please: Check if your dataset sampling rate and the parameter sample_rate here are matching, otherwise resample your audios
audio_config = VitsAudioConfig(
    sample_rate=22050,
    hop_length=256,
    win_length=1024,
    fft_size=1024,
    mel_fmin=0.0,
    mel_fmax=None,
    num_mels=80,
)

# Init VITSArgs setting the arguments that is needed for the YourTTS model
model_args = VitsArgs(
    d_vector_file=D_VECTOR_FILES,
    use_d_vector_file=True,
    d_vector_dim=512,
    num_layers_text_encoder=10,
    # usefull parameters to the enable multilingual training
    # use_language_embedding=True,
    # embedded_language_dim=4,
)

# General training config, here you can change the batch size and others usefull parameters
config = VitsConfig(
    output_path=OUT_PATH,
    model_args=model_args,
    run_name=RUN_NAME,
    project_name="YourTTS",
    run_description="""
            - Original YourTTS trained using VCTK dataset
        """,
    dashboard_logger="tensorboard",
    logger_uri=None,
    audio=audio_config,
    batch_size=BATCH_SIZE,
    batch_group_size=48,
    eval_batch_size=BATCH_SIZE,
    num_loader_workers=8,
    eval_split_max_size=256,
    print_step=50,
    plot_step=100,
    log_model_step=1000,
    save_step=5000,
    save_n_checkpoints=2,
    save_checkpoints=True,
    target_loss="loss_1",
    print_eval=False,
    use_phonemes=False,
    phonemizer="espeak",
    phoneme_language="en",
    compute_input_seq_cache=True,
    add_blank=True,
    text_cleaner="english_cleaners",
    phoneme_cache_path=None,
    precompute_num_workers=12,
    start_by_longest=True,
    datasets=datasets_list,
    cudnn_benchmark=False,
    max_audio_len=220500,  # it should be: sampling rate * max audio in sec. So it is 22050 * 10 = 220500
    mixed_precision=False,
    test_sentences=[
        [
            "It took me quite a long time to develop a voice, and now that I have it I'm not going to be silent.",
            "VCTK_p277",
            None,
            "en",
        ],
        [
            "Be a voice, not an echo.",
            "VCTK_p239",
            None,
            "en",
        ],
        [
            "I'm sorry Dave. I'm afraid I can't do that.",
            "VCTK_p258",
            None,
            "en",
        ],
        [
            "This cake is great. It's so delicious and moist.",
            "VCTK_p244",
            None,
            "en",
        ],
        [
            "Prior to November 22, 1963.",
            "VCTK_p305",
            None,
            "en",
        ],
    ],
    # Enable the weighted sampler
    use_weighted_sampler=True,
    # Ensures that all speakers are seen in the training batch equally no matter how many samples each speaker has
    weighted_sampler_attrs={"speaker_name": 1.0},
)

# Load all the datasets samples and split traning and evaluation sets
train_samples, eval_samples = load_tts_samples(
    config.datasets,
    eval_split=True,
    eval_split_max_size=config.eval_split_max_size,
    eval_split_size=config.eval_split_size,
)

# Init the model
model = Vits.init_from_config(config)

# Init the trainer and 🚀
trainer = Trainer(
    TrainerArgs(restore_path=RESTORE_PATH, skip_train_epoch=SKIP_TRAIN_EPOCH),
    config,
    output_path=OUT_PATH,
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples,
)
trainer.fit()
