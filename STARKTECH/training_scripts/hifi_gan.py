from TTS.trainer import Trainer, TrainingArgs, TensorboardLogger, TrainerCallback, init_training
from TTS.vocoder.configs.hifigan_config import HifiganConfig
from TTS.vocoder.models.gan import GAN


def example(name=None):
    if name == "model":
        args = TrainingArgs()
        config = HifiganConfig()
        model = GAN(config)
        trainer = Trainer(args, config, output_path="/home/logan", model=model)
        trainer.fit()
    elif name == "config":
        config = HifiganConfig(data_path="/home/erogol/nvme/gdrive/Datasets/LJSpeech-1.1/wavs/", output_path="/home/logan", )
        args, config, output_path, _, c_logger, tb_logger = init_training(TrainingArgs(), config)
        trainer = Trainer(args, config, output_path, c_logger, tb_logger)
        trainer.fit()
    else:
        print("please specify if you want to see a model example or a config example")


class Hifi_GAN:
    def __init__(self, batch_size, eval_batch_size, output_path, data_path):
        self.batch_size = batch_size
        self.output_path = output_path
        self.eval_batch_size = eval_batch_size
        self.data_path = data_path

    def single_speaker_model(self):
        pass

    def multi_speaker_vctk_model(self):
        args = TrainingArgs()
        config = HifiganConfig(batch_size=self.batch_size, eval_batch_size=self.eval_batch_size, data_path=self.data_path, use_noise_augment=False)
        model = GAN(config)
        trainer = Trainer(args, config, output_path=self.output_path, model=model)
        trainer.fit()

    def multi_speaker_libri_tts_model(self):
        pass

    def fine_tune_model(self):
        pass

    def config_model(self):
        pass
