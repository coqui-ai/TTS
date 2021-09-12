from TTS.auto_tts.complete_recipes import TtsAutoTrainer

trainer = TtsAutoTrainer(
    data_path="/media/logan/STARKTECH/VCTK-Corpus-removed-silence",\
    dataset="vctk",
    batch_size=32,
    learning_rate=0.001,
    mixed_precision=False,
    output_path="/home/logan/TTS/recipes/ljspeech/vits_tts",
    epochs=1000,
)

model = trainer.multi_speaker_autotts("vits tts", speaker_file="/home/logan/Downloads/tts_models--en--vctk--vits/speaker_ids.json", glowtts_encoder=None)

model.fit()
