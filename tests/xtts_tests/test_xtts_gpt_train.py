import os
import shutil

import torch
from trainer import Trainer, TrainerArgs

from tests import get_tests_output_path
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.layers.xtts.dvae import DiscreteVAE
from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer, GPTTrainerConfig, XttsAudioConfig

config_dataset = BaseDatasetConfig(
    formatter="ljspeech",
    dataset_name="ljspeech",
    path="tests/data/ljspeech/",
    meta_file_train="metadata.csv",
    meta_file_val="metadata.csv",
    language="en",
)

DATASETS_CONFIG_LIST = [config_dataset]

# Logging parameters
RUN_NAME = "GPT_XTTS_LJSpeech_FT"
PROJECT_NAME = "XTTS_trainer"
DASHBOARD_LOGGER = "tensorboard"
LOGGER_URI = None

# Set here the path that the checkpoints will be saved. Default: ./run/training/
OUT_PATH = os.path.join(get_tests_output_path(), "train_outputs", "xtts_tests")
os.makedirs(OUT_PATH, exist_ok=True)

# Create DVAE checkpoint and mel_norms on test time
# DVAE parameters: For the training we need the dvae to extract the dvae tokens, given that you must provide the paths for this model
DVAE_CHECKPOINT = os.path.join(OUT_PATH, "dvae.pth")  # DVAE checkpoint
MEL_NORM_FILE = os.path.join(
    OUT_PATH, "mel_stats.pth"
)  # Mel spectrogram norms, required for dvae mel spectrogram extraction
dvae = DiscreteVAE(
    channels=80,
    normalization=None,
    positional_dims=1,
    num_tokens=8192,
    codebook_dim=512,
    hidden_dim=512,
    num_resnet_blocks=3,
    kernel_size=3,
    num_layers=2,
    use_transposed_convs=False,
)
torch.save(dvae.state_dict(), DVAE_CHECKPOINT)
mel_stats = torch.ones(80)
torch.save(mel_stats, MEL_NORM_FILE)


# XTTS transfer learning parameters: You we need to provide the paths of XTTS model checkpoint that you want to do the fine tuning.
TOKENIZER_FILE = "tests/inputs/xtts_vocab.json"  # vocab.json file
XTTS_CHECKPOINT = None  # "/raid/edresson/dev/Checkpoints/XTTS_evaluation/xtts_style_emb_repetition_fix_gt/132500_gpt_ema_coqui_tts_with_enhanced_hifigan.pth"  # model.pth file


# Training sentences generations
SPEAKER_REFERENCE = [
    "tests/data/ljspeech/wavs/LJ001-0002.wav"
]  # speaker reference to be used in training test sentences
LANGUAGE = config_dataset.language


# Training Parameters
OPTIMIZER_WD_ONLY_ON_WEIGHTS = True  # for multi-gpu training please make it False
START_WITH_EVAL = False  # if True it will star with evaluation
BATCH_SIZE = 2  # set here the batch size
GRAD_ACUMM_STEPS = 1  # set here the grad accumulation steps
# Note: we recommend that BATCH_SIZE * GRAD_ACUMM_STEPS need to be at least 252 for more efficient training. You can increase/decrease BATCH_SIZE but then set GRAD_ACUMM_STEPS accordingly.


# init args and config
model_args = GPTArgs(
    max_conditioning_length=132300,  # 6 secs
    min_conditioning_length=66150,  # 3 secs
    debug_loading_failures=False,
    max_wav_length=255995,  # ~11.6 seconds
    max_text_length=200,
    mel_norm_file=MEL_NORM_FILE,
    dvae_checkpoint=DVAE_CHECKPOINT,
    xtts_checkpoint=XTTS_CHECKPOINT,  # checkpoint path of the model that you want to fine-tune
    tokenizer_file=TOKENIZER_FILE,
    gpt_num_audio_tokens=8194,
    gpt_start_audio_token=8192,
    gpt_stop_audio_token=8193,
)
audio_config = XttsAudioConfig(sample_rate=22050, dvae_sample_rate=22050, output_sample_rate=24000)
config = GPTTrainerConfig(
    epochs=1,
    output_path=OUT_PATH,
    model_args=model_args,
    run_name=RUN_NAME,
    project_name=PROJECT_NAME,
    run_description="""
        GPT XTTS training
        """,
    dashboard_logger=DASHBOARD_LOGGER,
    logger_uri=LOGGER_URI,
    audio=audio_config,
    batch_size=BATCH_SIZE,
    batch_group_size=48,
    eval_batch_size=BATCH_SIZE,
    num_loader_workers=8,
    eval_split_max_size=256,
    print_step=50,
    plot_step=100,
    log_model_step=1000,
    save_step=10000,
    save_n_checkpoints=1,
    save_checkpoints=True,
    # target_loss="loss",
    print_eval=False,
    # Optimizer values like tortoise, pytorch implementation with modifications to not apply WD to non-weight parameters.
    optimizer="AdamW",
    optimizer_wd_only_on_weights=OPTIMIZER_WD_ONLY_ON_WEIGHTS,
    optimizer_params={"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": 1e-2},
    lr=5e-06,  # learning rate
    lr_scheduler="MultiStepLR",
    # it was adjusted accordly for the new step scheme
    lr_scheduler_params={"milestones": [50000 * 18, 150000 * 18, 300000 * 18], "gamma": 0.5, "last_epoch": -1},
    test_sentences=[
        {
            "text": "This cake is great. It's so delicious and moist.",
            "speaker_wav": SPEAKER_REFERENCE,
            "language": LANGUAGE,
        },
    ],
)

# init the model from config
model = GPTTrainer.init_from_config(config)

# load training samples
train_samples, eval_samples = load_tts_samples(
    DATASETS_CONFIG_LIST,
    eval_split=True,
    eval_split_max_size=config.eval_split_max_size,
    eval_split_size=config.eval_split_size,
)

# init the trainer and 🚀
trainer = Trainer(
    TrainerArgs(
        restore_path=None,  # xtts checkpoint is restored via xtts_checkpoint key so no need of restore it using Trainer restore_path parameter
        skip_train_epoch=False,
        start_with_eval=True,
        grad_accum_steps=GRAD_ACUMM_STEPS,
    ),
    config,
    output_path=OUT_PATH,
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples,
)
trainer.fit()

# remove output path
shutil.rmtree(OUT_PATH)
