from TTS.utils.manage import ModelManager

manager = ModelManager()
model_path, config_path, x = manager.download_model("tts_models/en/ljspeech/tacotron2-DCA")

print(model_path)

print(config_path)

print(x)

manager.list_models()
