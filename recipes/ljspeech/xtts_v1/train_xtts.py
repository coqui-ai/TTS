from trainer import Trainer, TrainerArgs

from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer, GPTTrainerConfig, XttsAudioConfig

# Define here the dataset used
config_ljspeech = BaseDatasetConfig(
    formatter="ljspeech",
    dataset_name="ljspeech",
    path="/raid/datasets/LJSpeech-1.1_24khz/",
    meta_file_train="/raid/datasets/LJSpeech-1.1_24khz/metadata.csv",
    language="en",
)

DATASETS_CONFIG_LIST = [config_ljspeech]


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
        # tokenizer_file="/raid/datasets/xtts_models/vocab.json", # vocab path of the model that you want to fine-tune
        # xtts_checkpoint="https://huggingface.co/coqui/XTTS-v1/resolve/hifigan/model.pth",
        xtts_checkpoint="/raid/edresson/dev/Checkpoints/XTTS_evaluation/xtts_style_emb_repetition_fix_gt/132500_gpt_ema_coqui_tts_with_enhanced_hifigan.pth",  # checkpoint path of the model that you want to fine-tune
        tokenizer_file="/raid/edresson/dev/Checkpoints/XTTS_evaluation/xtts_style_emb_repetition_fix_gt/tokenizer_merged_5.json",
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
                "speaker_wav": "/raid/edresson/dev/ref-ljspeech.wav",
                "language": "en",
            },
            {
                "text": "This cake is great. It's so delicious and moist.",
                "speaker_wav": "/raid/edresson/dev/ref-ljspeech.wav",
                "language": "en",
            },
            {
                "text": "Levei muito tempo para desenvolver uma voz e agora que a tenho não vou ficar calado .",
                "speaker_wav": "/raid/edresson/dev/ref-ljspeech.wav",
                "language": "pt",
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
    RUN_NAME = "GPT_XTTS_LJSpeech_fixed"
    PROJECT_NAME = "XTTS_trainer"
    OUT_PATH = "/raid/edresson/dev/Checkpoints/XTTS_v1_FT/"
    # DASHBOARD_LOGGER = "clearml"
    # LOGGER_URI = "s3://coqui-ai-models/TTS/Checkpoints/XTTS_v1/"
    DASHBOARD_LOGGER = "tensorboard"
    LOGGER_URI = None
    RESTORE_PATH = None
    SKIP_TRAIN_EPOCH = False
    START_WITH_EVAL = True
    BATCH_SIZE = 3
    GRAD_ACUMM_STEPS = 28 * 3

    # debug
    # DASHBOARD_LOGGER = "tensorboard"
    # LOGGER_URI = None
    # RESTORE_PATH = None
    # BATCH_SIZE = 2
    # GRAD_ACUMM_STEPS = 1

    main()
