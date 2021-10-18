from TTS.auto_tts.complete_recipes import VocoderAutoTrainer, TtsAutoTrainer

trainer = TtsAutoTrainer(
    data_path="../LJSpeech-1.1",
    dataset="ljspeech",
    batch_size=32,
    learning_rate=[0.001, 0.001],
    mixed_precision=False,
    output_path="../TTS/recipes/ljspeech/vits_tts",
    epochs=1000,
)

model = trainer.from_pretrained("sc-glow-tts")

# model.fit()
