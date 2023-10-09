import os

import torch
from trainer import Trainer, TrainerArgs

from TTS.bin.compute_embeddings import compute_embeddings
from TTS.bin.resample import resample_files
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import CharactersConfig, Vits, VitsArgs, VitsAudioConfig
from TTS.utils.downloaders import download_vctk
from TTS.tts.utils.languages import LanguageManager
import test_sentences

torch.set_num_threads(24)

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
MANIFEST_FOLDER = "<PATH TO MANIFEST FILES>"
with open("<PATH TO CHARECTER SET>", "r") as f:
    charecter_set = f.read().strip("\n")

OUT_PATH = os.path.dirname(os.path.abspath(__file__))  

BATCH_SIZE = 32
SAMPLE_RATE = 16000

DATASETS_CONFIG_LIST = [
    BaseDatasetConfig(
    formatter="syspin_ml",
    meta_file_train=os.path.join(MANIFEST_FOLDER, fname),
    path="model_related/" + fname.split(".")[0],
    language=fname.split("_")[0]
) for fname in os.listdir(MANIFEST_FOLDER)
    ]

SPEAKER_ENCODER_CHECKPOINT_PATH = (
    "https://github.com/coqui-ai/TTS/releases/download/speaker_encoder_model/model_se.pth.tar"
)
SPEAKER_ENCODER_CONFIG_PATH = "https://github.com/coqui-ai/TTS/releases/download/speaker_encoder_model/config_se.json"

D_VECTOR_FILES = []  

for dataset_conf in DATASETS_CONFIG_LIST:
    embeddings_file = os.path.join(dataset_conf.path, "speakers.pth")
    if not os.path.isfile(embeddings_file):
        print(f">>> Computing the speaker embeddings for the {dataset_conf.dataset_name} dataset")
        compute_embeddings(
            SPEAKER_ENCODER_CHECKPOINT_PATH,
            SPEAKER_ENCODER_CONFIG_PATH,
            embeddings_file,
            old_speakers_file=None,
            config_dataset_path=None,
            formatter_name=dataset_conf.formatter,
            dataset_name=dataset_conf.dataset_name,
            dataset_path=dataset_conf.path,
            meta_file_train=dataset_conf.meta_file_train,
            meta_file_val=dataset_conf.meta_file_val,
            disable_cuda=False,
            no_eval=False,
        )
    D_VECTOR_FILES.append(embeddings_file)

audio_config = VitsAudioConfig(
    sample_rate=SAMPLE_RATE,
    hop_length=256,
    win_length=1024,
    fft_size=1024,
    mel_fmin=0.0,
    mel_fmax=None,
    num_mels=80,
)

model_args = VitsArgs(
    d_vector_file=D_VECTOR_FILES,
    use_language_embedding=True,
    embedded_language_dim=4,
    use_d_vector_file=True,
    d_vector_dim=512,
    num_layers_text_encoder=10,
    speaker_encoder_model_path=SPEAKER_ENCODER_CHECKPOINT_PATH,
    speaker_encoder_config_path=SPEAKER_ENCODER_CONFIG_PATH,
    resblock_type_decoder="2",  
)

config = VitsConfig(
    output_path=OUT_PATH,
    model_args=model_args,
    run_name="yourtts_syspin_14spk_basemodel_1hr_per_spk",
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
    text_cleaner="multilingual_cleaners",
    characters=CharactersConfig(
        characters_class="TTS.tts.models.vits.VitsCharacters",
        pad="_",
        eos="&",
        bos="*",
        blank=None,
        characters=charecter_set,
        punctuations="!'(),-.:;? ",
        phonemes="",
        is_unique=True,
        is_sorted=True,
    ),
    phoneme_cache_path=None,
    precompute_num_workers=12,
    start_by_longest=True,
    datasets=DATASETS_CONFIG_LIST,
    cudnn_benchmark=False,
    mixed_precision=False,
    test_sentences=test_sentences.sents[0],
    use_weighted_sampler=True,
    speaker_encoder_loss_alpha=9.0,
)

train_samples, eval_samples = load_tts_samples(
    config.datasets,
    eval_split=True,
    eval_split_max_size=config.eval_split_max_size,
    eval_split_size=config.eval_split_size,
)

language_manager = LanguageManager(config=config)
config.model_args.num_languages = language_manager.num_languages

model = Vits.init_from_config(config)

trainer = Trainer(
    TrainerArgs(restore_path=None, skip_train_epoch=False),
    config,
    output_path=OUT_PATH,
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples,
)
trainer.fit()
