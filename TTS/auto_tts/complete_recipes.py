from TTS.auto_tts.model_hub import Models
from TTS.auto_tts.utils import data_loader
from TTS.trainer import Trainer, TrainingArgs, init_training


class Examples:
    """This is trainer for calling complete recipes based off public datasets.
    all configs are based off pretrained model configs or the model papers.

    usage:
            From TTS.auto_tts.complete_recipes import TtsTrainer
            trainer = Examples(data_path='DEFINE THIS', batch_size=32, learning_rate=0.001,
                      mixed_precision=False, output_path='DEFINE THIS', epochs=1000)
            model = trainer.ljspeech_tacotron2("double decoder consistency")
            model.fit()
    """

    def __init__(self, data_path, batch_size, output_path, mixed_precision, learning_rate, epochs):

        self.data_path = data_path
        self.batch_size = batch_size
        self.output_path = output_path
        self.mixed_precision = mixed_precision
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.model = Models(
            batch_size=self.batch_size,
            mixed_precision=self.mixed_precision,
            learning_rate=self.learning_rate,
            epochs=self.epochs,
        )

    def ljspeech_tacotron2(self, name="tacotron2"):
        if name == "double decoder consistency":
            dataset, audio = data_loader(
                name="ljspeech", path=self.data_path, stats_path="stats_path/scale_stats_ddc.npy"
            )
            model_config = self.model.single_speaker_tacotron2_DDC(audio, dataset)
        elif name == "dynamic convolution attention":
            dataset, audio = data_loader(
                name="ljspeech", path=self.data_path, stats_path="stats_path/scale_stats_dca.npy"
            )
            model_config = self.model.single_speaker_tacotron2_DCA(audio, dataset)
        elif name == "tacotron2":
            dataset, audio = data_loader(name="ljspeech", path=self.data_path, stats_path=None)
            model_config = self.model.single_speaker_tacotron2_base(audio, dataset)
        args, config, output_path, _, c_logger, tb_logger = init_training(TrainingArgs(), model_config)
        trainer = Trainer(args, config, output_path, c_logger, tb_logger)
        return trainer

    def ljspeech_glowtts(self):
        dataset, audio = data_loader(name="ljspeech", path=self.data_path)
        model_config = self.model.ljspeech_glow_tts(audio, dataset)
        args, config, output_path, _, c_logger, tb_logger = init_training(TrainingArgs(), model_config)
        trainer = Trainer(args, config, output_path, c_logger, tb_logger)
        return trainer

    def ljspeech_speedy_speech(self):
        dataset, audio = data_loader(name="ljspeech", path=self.data_path)
        model_config = self.model.ljspeech_speedy_speech(audio, dataset)
        args, config, output_path, _, c_logger, tb_logger = init_training(TrainingArgs(), model_config)
        trainer = Trainer(args, config, output_path, c_logger, tb_logger)
        return trainer
