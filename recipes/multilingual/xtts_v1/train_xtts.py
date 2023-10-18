from trainer import Trainer, TrainerArgs

from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer, GPTTrainerConfig, XttsAudioConfig

config_coqui_MLS_metadata_train_with_previous_audio_key_de = BaseDatasetConfig(
    formatter="coqui",
    dataset_name="coqui",
    path="/raid/datasets/MLS/mls_german",
    meta_file_train="metadata_train_with_previous_audio_key.csv",
    language="de",
)


config_coqui_MLS_metadata_test_with_previous_audio_key_de = BaseDatasetConfig(
    formatter="coqui",
    dataset_name="coqui",
    path="/raid/datasets/MLS/mls_german",
    meta_file_train="metadata_test_with_previous_audio_key.csv",
    language="de",
)


config_coqui_MLS_metadata_dev_with_previous_audio_key_de = BaseDatasetConfig(
    formatter="coqui",
    dataset_name="coqui",
    path="/raid/datasets/MLS/mls_german",
    meta_file_train="metadata_dev_with_previous_audio_key.csv",
    language="de",
)


config_coqui_mls_french_metadata_with_previous_audio_key_fr = BaseDatasetConfig(
    formatter="coqui",
    dataset_name="coqui",
    path="/raid/datasets/MLS/mls_french/",
    meta_file_train="metadata_with_previous_audio_key.csv",
    language="fr",
)


config_coqui_mls_spanish_metadata_with_previous_audio_key_es = BaseDatasetConfig(
    formatter="coqui",
    dataset_name="coqui",
    path="/raid/datasets/MLS/mls_spanish/",
    meta_file_train="/raid/datasets/MLS/mls_spanish/metadata_with_previous_audio_key.csv",
    language="es",
)


config_coqui_mls_italian_metadata_with_previous_audio_key_it = BaseDatasetConfig(
    formatter="coqui",
    dataset_name="coqui",
    path="/raid/datasets/MLS/mls_italian/",
    meta_file_train="/raid/datasets/MLS/mls_italian/metadata_with_previous_audio_key.csv",
    language="it",
)


config_coqui_mls_portuguese_metadata_with_previous_audio_key_pt = BaseDatasetConfig(
    formatter="coqui",
    dataset_name="coqui",
    path="/raid/datasets/MLS/mls_portuguese/",
    meta_file_train="/raid/datasets/MLS/mls_portuguese/metadata_with_previous_audio_key.csv",
    language="pt",
)


config_coqui_mls_polish_metadata_with_previous_audio_key_pl = BaseDatasetConfig(
    formatter="coqui",
    dataset_name="coqui",
    path="/raid/datasets/MLS/mls_polish/",
    meta_file_train="/raid/datasets/MLS/mls_polish/metadata_with_previous_audio_key.csv",
    language="pl",
)


config_coqui_common_voice_metafile_it_train_with_scores_it = BaseDatasetConfig(
    formatter="coqui",
    dataset_name="coqui",
    path="/raid/datasets/common_voice/",
    meta_file_train="/raid/datasets/common_voice/metafile_it_train_with_scores.csv",
    language="it",
)


config_coqui_common_voice_metafile_it_test_with_scores_it = BaseDatasetConfig(
    formatter="coqui",
    dataset_name="coqui",
    path="/raid/datasets/common_voice/",
    meta_file_train="/raid/datasets/common_voice/metafile_it_test_with_scores.csv",
    language="it",
)


config_coqui_common_voice_metafile_it_dev_with_scores_it = BaseDatasetConfig(
    formatter="coqui",
    dataset_name="coqui",
    path="/raid/datasets/common_voice/",
    meta_file_train="/raid/datasets/common_voice/metafile_it_dev_with_scores.csv",
    language="it",
)


config_coqui_common_voice_metafile_pt_train_with_scores_pt = BaseDatasetConfig(
    formatter="coqui",
    dataset_name="coqui",
    path="/raid/datasets/common_voice/",
    meta_file_train="/raid/datasets/common_voice/metafile_pt_train_with_scores.csv",
    language="pt",
)


config_coqui_common_voice_metafile_pt_test_with_scores_pt = BaseDatasetConfig(
    formatter="coqui",
    dataset_name="coqui",
    path="/raid/datasets/common_voice/",
    meta_file_train="/raid/datasets/common_voice/metafile_pt_test_with_scores.csv",
    language="pt",
)


config_coqui_common_voice_metafile_pt_dev_with_scores_pt = BaseDatasetConfig(
    formatter="coqui",
    dataset_name="coqui",
    path="/raid/datasets/common_voice/",
    meta_file_train="/raid/datasets/common_voice/metafile_pt_dev_with_scores.csv",
    language="pt",
)


config_coqui_common_voice_metafile_en_train_en = BaseDatasetConfig(
    formatter="coqui",
    dataset_name="coqui",
    path="/raid/datasets/common_voice/",
    meta_file_train="/raid/datasets/common_voice/metafile_en_train.csv",
    language="en",
)


config_coqui_common_voice_metafile_en_test_en = BaseDatasetConfig(
    formatter="coqui",
    dataset_name="coqui",
    path="/raid/datasets/common_voice/",
    meta_file_train="/raid/datasets/common_voice/metafile_en_test.csv",
    language="en",
)


config_coqui_common_voice_metafile_en_dev_en = BaseDatasetConfig(
    formatter="coqui",
    dataset_name="coqui",
    path="/raid/datasets/common_voice/",
    meta_file_train="/raid/datasets/common_voice/metafile_en_dev.csv",
    language="en",
)


config_coqui_common_voice_metafile_tr_validated_tr = BaseDatasetConfig(
    formatter="coqui",
    dataset_name="coqui",
    path="/raid/datasets/common_voice/",
    meta_file_train="/raid/datasets/common_voice/metafile_tr_validated.csv",
    language="tr",
)


config_coqui_common_voice_metafile_ru_validated_ru = BaseDatasetConfig(
    formatter="coqui",
    dataset_name="coqui",
    path="/raid/datasets/common_voice/",
    meta_file_train="/raid/datasets/common_voice/metafile_ru_validated.csv",
    language="ru",
)


config_coqui_common_voice_metafile_nl_validated_nl = BaseDatasetConfig(
    formatter="coqui",
    dataset_name="coqui",
    path="/raid/datasets/common_voice/",
    meta_file_train="/raid/datasets/common_voice/metafile_nl_validated.csv",
    language="nl",
)


config_coqui_common_voice_metafile_cs_validated_cs = BaseDatasetConfig(
    formatter="coqui",
    dataset_name="coqui",
    path="/raid/datasets/common_voice/",
    meta_file_train="/raid/datasets/common_voice/metafile_cs_validated.csv",
    language="cs",
)


config_coqui_common_voice_metafile_fr_validated_fr = BaseDatasetConfig(
    formatter="coqui",
    dataset_name="coqui",
    path="/raid/datasets/common_voice/",
    meta_file_train="/raid/datasets/common_voice/metafile_fr_validated.csv",
    language="fr",
)


config_coqui_common_voice_metafile_es_validated_es = BaseDatasetConfig(
    formatter="coqui",
    dataset_name="coqui",
    path="/raid/datasets/common_voice/",
    meta_file_train="/raid/datasets/common_voice/metafile_es_validated.csv",
    language="es",
)


config_coqui_common_voice_metafile_pl_validated_pl = BaseDatasetConfig(
    formatter="coqui",
    dataset_name="coqui",
    path="/raid/datasets/common_voice/",
    meta_file_train="/raid/datasets/common_voice/metafile_pl_validated.csv",
    language="pl",
)


config_coqui_common_voice_metafile_ar_validated_ar = BaseDatasetConfig(
    formatter="coqui",
    dataset_name="coqui",
    path="/raid/datasets/common_voice/",
    meta_file_train="/raid/datasets/common_voice/metafile_ar_validated.csv",
    language="ar",
)


config_coqui_common_voice_metafile_zh_CN_validated_zh_cn = BaseDatasetConfig(
    formatter="coqui",
    dataset_name="coqui",
    path="/raid/datasets/common_voice/",
    meta_file_train="/raid/datasets/common_voice/metafile_zh-CN_validated.csv",
    language="zh-cn",
)


config_coqui_common_voice_metafile_ja_validated_ja = BaseDatasetConfig(
    formatter="coqui",
    dataset_name="coqui",
    path="/raid/datasets/common_voice/",
    meta_file_train="/raid/datasets/common_voice/metafile_ja_validated.csv",
    language="ja",
)

# DATASETS_CONFIG_LIST=[config_coqui_MLS_metadata_train_with_previous_audio_key_de, config_coqui_MLS_metadata_test_with_previous_audio_key_de, config_coqui_MLS_metadata_dev_with_previous_audio_key_de, config_coqui_mls_french_metadata_with_previous_audio_key_fr, config_coqui_mls_spanish_metadata_with_previous_audio_key_es, config_coqui_mls_italian_metadata_with_previous_audio_key_it, config_coqui_mls_portuguese_metadata_with_previous_audio_key_pt, config_coqui_mls_polish_metadata_with_previous_audio_key_pl, config_coqui_common_voice_metafile_it_train_with_scores_it, config_coqui_common_voice_metafile_it_test_with_scores_it, config_coqui_common_voice_metafile_it_dev_with_scores_it, config_coqui_common_voice_metafile_pt_train_with_scores_pt, config_coqui_common_voice_metafile_pt_test_with_scores_pt, config_coqui_common_voice_metafile_pt_dev_with_scores_pt, config_coqui_common_voice_metafile_en_train_en, config_coqui_common_voice_metafile_en_test_en, config_coqui_common_voice_metafile_en_dev_en, config_coqui_common_voice_metafile_tr_validated_tr, config_coqui_common_voice_metafile_ru_validated_ru, config_coqui_common_voice_metafile_nl_validated_nl, config_coqui_common_voice_metafile_cs_validated_cs, config_coqui_common_voice_metafile_fr_validated_fr, config_coqui_common_voice_metafile_es_validated_es, config_coqui_common_voice_metafile_pl_validated_pl, config_coqui_common_voice_metafile_ar_validated_ar, config_coqui_common_voice_metafile_zh_CN_validated_zh_cn, config_coqui_common_voice_metafile_ja_validated_ja]

# DATASETS_CONFIG_LIST = [config_coqui_mls_french_metadata_with_previous_audio_key_fr, config_coqui_MLS_metadata_test_with_previous_audio_key_de, config_coqui_mls_spanish_metadata_with_previous_audio_key_es, config_coqui_mls_italian_metadata_with_previous_audio_key_it]

DATASETS_CONFIG_LIST = [
    config_coqui_MLS_metadata_test_with_previous_audio_key_de,
    config_coqui_mls_italian_metadata_with_previous_audio_key_it,
]


def freeze_layers(trainer):
    pass


def main():
    # init args and config
    model_args = GPTArgs(
        max_conditioning_length=132300,  # 6 secs
        min_conditioning_length=66150,  # 3 secs
        debug_loading_failures=False,
        max_wav_length=255995,  # ~11.6 seconds
        max_text_length=200,
        mel_norm_file="/raid/datasets/xtts_models/mel_stats.pth",
        dvae_checkpoint="/raid/datasets/xtts_models/dvae.pth",
        tokenizer_file="/raid/datasets/xtts_models/vocab.json",  # vocab path of the model that you want to fine-tune
        xtts_checkpoint="https://huggingface.co/coqui/XTTS-v1/resolve/hifigan/model.pth",  # checkpoint path of the model that you want to fine-tune
        gpt_num_audio_tokens=8194,
        gpt_start_audio_token=8192,
        gpt_stop_audio_token=8193,
    )
    audio_config = XttsAudioConfig(
        sample_rate=22050, dvae_sample_rate=22050, diffusion_sample_rate=24000, output_sample_rate=24000  # GPT SR
    )
    config = GPTTrainerConfig(
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
        optimizer_wd_only_on_weights=True,  # for multi-gpu training turn it off
        optimizer_params={"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": 1e-2},
        lr=5e-06,  # learning rate
        lr_scheduler="MultiStepLR",
        # it was adjusted accordly for the new step scheme
        lr_scheduler_params={"milestones": [50000 * 18, 150000 * 18, 300000 * 18], "gamma": 0.5, "last_epoch": -1},
        test_sentences=[
            {
                "text": "It took me quite a long time to develop a voice, and now that I have it I'm not going to be silent.",
                "speaker_wav": "/raid/edresson/dev/ref.wav",
                "language": "en",
            },
            {
                "text": "This cake is great. It's so delicious and moist.",
                "speaker_wav": "/raid/edresson/dev/ref.wav",
                "language": "en",
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
            restore_path=RESTORE_PATH,
            skip_train_epoch=SKIP_TRAIN_EPOCH,
            start_with_eval=START_WITH_EVAL,
            grad_accum_steps=GRAD_ACUMM_STEPS,
        ),
        config,
        output_path=OUT_PATH,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
        callbacks={"on_epoch_start": freeze_layers},
    )
    trainer.fit()


if __name__ == "__main__":
    RUN_NAME = "GPT_XTTS"
    PROJECT_NAME = "XTTS_trainer"
    OUT_PATH = "/raid/edresson/dev/Checkpoints/XTTS_style_emb/"
    DASHBOARD_LOGGER = "clearml"
    LOGGER_URI = "s3://coqui-ai-models/TTS/Checkpoints/XTTS_style_emb/"
    RESTORE_PATH = None
    SKIP_TRAIN_EPOCH = False
    START_WITH_EVAL = True
    BATCH_SIZE = 9
    GRAD_ACUMM_STEPS = 28

    # debug
    # DASHBOARD_LOGGER = "tensorboard"
    # LOGGER_URI = None
    # RESTORE_PATH = None
    BATCH_SIZE = 2
    GRAD_ACUMM_STEPS = 1

    main()
