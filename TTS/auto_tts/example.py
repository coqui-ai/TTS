from TTS.auto_tts.complete_recipes import Examples

trainer = Examples(
    data_path="DEFINE THIS",
    batch_size=32,
    learning_rate=0.001,
    mixed_precision=False,
    output_path="DEFINE THIS",
    epochs=1000,
)

model = trainer.ljspeech_tacotron2("double decoder consistency")

model.fit()
