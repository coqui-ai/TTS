#-*- coding: utf-8 -*-
import os

from trainer import Trainer, TrainerArgs
from TTS.tts.models.vits import CharactersConfig
from TTS.config.shared_configs import BaseAudioConfig
from TTS.tts.configs.glow_tts_config import GlowTTSConfig
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.glow_tts import GlowTTS
from TTS.tts.utils.speakers import SpeakerManager
from TTS.tts.utils.languages import LanguageManager
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor



if __name__=="__main__":
    # set experiment paths
    output_path = "C:\\Users\\82109\\Desktop\\TTS\\data\\vits"
    dataset_path = ["C:\\Users\\82109\\Desktop\\TTS\\data\\emilia", "C:\\Users\\82109\\Desktop\\TTS\\data\\kss"]
    #b_input_ids = torch.tensor(b_input_ids).to(device).long()
    dataset_config = [
        BaseDatasetConfig(name=path.split("\\")[-1], meta_file_train="text.json", path=path)
        for path in dataset_path
    ]

    # download the dataset if not downloaded

    # define dataset config

    # define audio config
    # ❗ resample the dataset externally using `TTS/bin/resample.py` and set `resample=False` for faster training
    audio_config = BaseAudioConfig(sample_rate=22050, resample=False, do_trim_silence=True, trim_db=23.0)

    # define model config
    config = GlowTTSConfig(
        batch_size=16,
        eval_batch_size=8,
        num_loader_workers=4,
        num_eval_loader_workers=4,
        precompute_num_workers=4,
        run_eval=True,
        test_delay_epochs=-1,
        epochs=1000,
        text_cleaner="korean_cleaners",
        use_phonemes=True,
        phoneme_language='ko-kr',
        phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),
        print_step=25,
        print_eval=False,
        mixed_precision=True,
        output_path=output_path,
        datasets=dataset_config,
        use_speaker_embedding=True,
        min_text_len=0,
        max_text_len=500,
        min_audio_len=0,
        max_audio_len=500000,
        characters=CharactersConfig(
            characters_class= "TTS.tts.utils.text.characters.Hangeul",

        ),
        test_sentences=[
            "이 음성은 음성합성으로 만들어진 음성입니다."
        ],
        add_blank=False,
        enable_eos_bos_chars=False
    )


    # INITIALIZE THE AUDIO PROCESSOR
    # Audio processor is used for feature extraction and audio I/O.
    # It mainly serves to the dataloader and the training loggers.
    ap = AudioProcessor.init_from_config(config)
    print(config.test_sentences)
    # INITIALIZE THE TOKENIZER
    # Tokenizer is used to convert text to sequences of token IDs.
    # If characters are not defined in the config, default characters are passed to the config
    tokenizer, config = TTSTokenizer.init_from_config(config)

    # LOAD DATA SAMPLES
    # Each sample is a list of ```[text, audio_file_path, speaker_name]```
    # You can define your custom sample loader returning the list of samples.
    # Or define your custom formatter and pass it to the `load_tts_samples`.
    # Check `TTS.tts.datasets.load_tts_samples` for more details.
    train_samples, eval_samples = load_tts_samples(
        dataset_config,
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
    )

    # init speaker manager for multi-speaker training
    # it maps speaker-id to speaker-name in the model and data-loader
    speaker_manager = SpeakerManager()
    speaker_manager.set_ids_from_data(train_samples + eval_samples, parse_key="speaker_name")
    config.num_speakers = speaker_manager.num_speakers



    # init model
    model = GlowTTS(config, ap, tokenizer, speaker_manager=speaker_manager)

    # INITIALIZE THE TRAINER
    # Trainer provides a generic API to train all the 🐸TTS models with all its perks like mixed-precision training,
    # distributed training, etc.
    trainer = Trainer(
        TrainerArgs(), config, output_path, model=model, train_samples=train_samples, eval_samples=eval_samples
    )

    # AND... 3,2,1... 🚀
    trainer.fit()