import os
from glob import glob

from trainer import Trainer, TrainerArgs

from TTS.config.shared_configs import BaseAudioConfig
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import CharactersConfig, Vits, VitsArgs
from TTS.tts.utils.languages import LanguageManager
from TTS.tts.utils.speakers import SpeakerManager
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor
import torch
from TTS.tts.datasets import jsonfomatter


if __name__ == '__main__':
    output_path = "C:\\Users\\82109\\Desktop\\TTS\\data\\vits"
    dataset_paths = ["C:\\Users\\82109\\Desktop\\TTS\\data\\kss", "C:\\Users\\82109\\Desktop\\TTS\\data\\emilia"]

    dataset_config = [
            BaseDatasetConfig(name=path.split("\\")[-1], meta_file_train="text.json", path=path,language="ko-kr")
            for path in dataset_paths
        ]


    #dataset_config=[]
    dataset_config.append(BaseDatasetConfig(name='jsut', meta_file_train="jtext.json", path='C:\\Users\\82109\\Desktop\\TTS\\data\\jsut',language="ja-jp"))

    audio_config = BaseAudioConfig(
        sample_rate=22050,
        win_length=1024,
        hop_length=256,
        num_mels=80,
        preemphasis=0.0,
        ref_level_db=20,
        log_func="np.log",
        do_trim_silence=False,
        trim_db=23.0,
        mel_fmin=0,
        mel_fmax=None,
        spec_gain=1.0,
        signal_norm=True,
        do_amp_to_db_linear=False,
        resample=False,
    )

    vitsArgs = VitsArgs(
        use_language_embedding=True,
        embedded_language_dim=4,
        use_speaker_embedding=True,
        use_sdp=False,
    )

    config = VitsConfig(
        model_args=vitsArgs,
        audio=audio_config,
        run_name="ejk",
        use_speaker_embedding=True,
        use_language_embedding=True,
        batch_size=4,
        eval_batch_size=2,
        batch_group_size=0,
        num_loader_workers=4,
        num_eval_loader_workers=4,
        run_eval=True,
        test_delay_epochs=-1,
        epochs=1000,
        text_cleaner="basic_cleaners",
        use_phonemes=True,
        phonemizer='multi_phonemizer',
        phoneme_language='multi-lingual',
        phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),
        compute_input_seq_cache=True,
        print_step=25,
        use_language_weighted_sampler=True,
        print_eval=False,
        mixed_precision=False,
        sort_by_audio_len=True,
        min_audio_len=32 * 256 * 4,
        max_audio_len=160000,
        output_path=output_path,
        datasets=dataset_config,
        characters=CharactersConfig(
            characters_class= "TTS.tts.utils.text.characters.Hangeul",
        ),
        test_sentences=[
            ["今なら許してあげる。だから潔く盗んだものを返して。", "emilia", None, "ja-jp"]
        ],
        add_blank=False,
        num_speakers=3,










    )

    # force the convertion of the custom characters to a config attribute
    config.from_dict(config.to_dict())

    # init audio processor
    ap = AudioProcessor(**config.audio.to_dict())

    # load training samples
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
    config.model_args.num_speakers = speaker_manager.num_speakers
    print(speaker_manager.ids)


    language_manager = LanguageManager(config=config)
    config.model_args.num_languages = language_manager.num_languages



    # INITIALIZE THE TOKENIZER
    # Tokenizer is used to convert text to sequences of token IDs.
    # config is updated with the default characters if not defined in the config.
    tokenizer, config = TTSTokenizer.init_from_config(config)

    # init model
    model = Vits(config, ap, tokenizer,speaker_manager, language_manager=language_manager)

    # init the trainer and 🚀
    trainer = Trainer(
        TrainerArgs(), config, output_path, model=model, train_samples=train_samples, eval_samples=eval_samples
    )
  #print(torch.embedding())
    trainer.fit()
