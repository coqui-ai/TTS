from TTS.tts.layers.losses import *


def setup_loss(config):
    if config.model.lower() in ["tacotron", "tacotron2"]:
        model = TacotronLoss(config)
    elif config.model.lower() == "glow_tts":
        model = GlowTTSLoss()
    elif config.model.lower() == "speedy_speech":
        model = SpeedySpeechLoss(config)
    elif config.model.lower() == "align_tts":
        model = AlignTTSLoss(config)
    else:
        raise ValueError(f" [!] loss for model {config.model.lower()} cannot be found.")
    return model
