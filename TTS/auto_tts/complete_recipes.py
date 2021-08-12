from TTS.auto_tts.model_hub import TtsModels, VocoderModels
from TTS.auto_tts.utils import data_loader
from TTS.trainer import Trainer, TrainingArgs, init_training


class TtsExamples:
    """This is trainer for calling complete recipes based off public datasets.
    all configs are based off pretrained model configs or the model papers.

    usage:
            From TTS.auto_tts.complete_recipes import TtsTrainer
            trainer = TtsExamples(data_path='DEFINE THIS', batch_size=32, learning_rate=0.001,
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
        self.model = TtsModels(
            batch_size=self.batch_size,
            mixed_precision=self.mixed_precision,
            learning_rate=self.learning_rate,
            epochs=self.epochs,
        )

    def ljspeech_tacotron2(self, name="tacotron2", forward_attention=False, location_attention=True):
        if name == "double decoder consistency":
            dataset, audio = data_loader(
                name="ljspeech", path=self.data_path, stats_path="stats_path/scale_stats_ddc.npy"
            )
            model_config = self.model.single_speaker_tacotron2_DDC(
                audio, dataset, forward_attn=forward_attention, location_attn=location_attention
            )
        elif name == "dynamic convolution attention":
            dataset, audio = data_loader(
                name="ljspeech", path=self.data_path, stats_path="stats_path/scale_stats_dca.npy"
            )
            model_config = self.model.single_speaker_tacotron2_DCA(
                audio, dataset, forward_attn=forward_attention, location_attn=location_attention
            )
        elif name == "tacotron2":
            dataset, audio = data_loader(name="ljspeech", path=self.data_path, stats_path=None)
            model_config = self.model.single_speaker_tacotron2_base(
                audio, dataset, forward_attn=forward_attention, location_attn=location_attention
            )
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

    def vctk_glow_tts(self, speaker_file, encoder="transformer"):
        dataset, audio = data_loader(name="vctk", path=self.data_path)
        model_config = self.model.ScGlowTts(audio, dataset, speaker_file, encoder=encoder)
        args, config, output_path, _, c_logger, tb_logger = init_training(TrainingArgs(), model_config)
        trainer = Trainer(args, config, output_path, c_logger, tb_logger)
        return trainer

    def sam_accenture_tacotron2(
        self, name="double decoder consistency", forward_attention=False, location_attention=True
    ):
        """Tacotron2 recipes for the sam dataset, based off the pre trained model."""
        dataset, audio = data_loader(name="sam", path=self.data_path)
        if name == "double decoder consistency":
            model_config = self.model.single_speaker_tacotron2_DDC(
                audio,
                dataset,
                pla=0.5,
                dla=0.5,
                ga=0.0,
                forward_attn=forward_attention,
                location_attn=location_attention,
            )
        elif name == "dynamic convolution attention":
            model_config = self.model.single_speaker_tacotron2_DCA(
                audio,
                dataset,
                pla=0.5,
                dla=0.5,
                ga=0.0,
                forward_attn=forward_attention,
                location_attn=location_attention,
            )
        args, config, output_path, _, c_logger, tb_logger = init_training(TrainingArgs(), model_config)
        trainer = Trainer(args, config, output_path, c_logger, tb_logger)
        return trainer


class VocoderExamples:
    """This is the class that will hold all the vocoder recipes,
    decided to split the tts recipes and vocoder recipes because it just makes sense to me."""

    def __init__(self, data_path, batch_size, output_path, mixed_precision, learning_rate, epochs):

        self.data_path = data_path
        self.batch_size = batch_size
        self.output_path = output_path
        self.mixed_precision = mixed_precision
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.model = VocoderModels(
            batch_size=self.batch_size,
            mixed_precision=self.mixed_precision,
            learning_rate=self.learning_rate,
            epochs=self.epochs,
        )

    def ljspeech_vocoder(self, name=None):
        dataset, audio = data_loader(name="ljspeech", path=self.data_path, stats_path="")
        if name == "hifigan":
            model_config = self.model.ljspeech_hifigan(audio, self.data_path)
        elif name == "wavegrad":
            model_config = self.model.ljspeech_hifigan(audio, self.data_path)
        args, config, output_path, _, c_logger, tb_logger = init_training(TrainingArgs(), model_config)
        trainer = Trainer(args, config, output_path, c_logger, tb_logger)
        return trainer
