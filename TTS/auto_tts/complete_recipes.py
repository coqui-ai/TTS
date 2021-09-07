from TTS.auto_tts.model_hub import TtsModels, VocoderModels
from TTS.auto_tts.utils import data_loader
from TTS.trainer import Trainer, TrainingArgs, init_training
from recipes.ljspeech.glow_tts import train_glowtts
from recipes.ljpeech.vits_tts import train_vits
from recipes.ljpseech


class TtsAutoTrainer:
    """This is trainer for calling complete recipes based off public datasets.
    all configs are based off pretrained model configs or the model papers.

    Examples:
            From TTS.auto_tts.complete_recipes import TtsAutoTrainer
            trainer = TtsExamples(data_path='DEFINE THIS', batch_size=32, learning_rate=0.001,
                      mixed_precision=False, output_path='DEFINE THIS', epochs=1000)
            model = trainer.ljspeechAutoTts("tacotron2, "double decoder consistency")
            model.fit()
    """

    def __init__(self, data_path, dataset, batch_size, output_path, mixed_precision, learning_rate, epochs):

        self.data_path = data_path
        self.dataset_name = dataset
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

    def single_speaker_autotts(
        self,
        model_name,
        stats_path=None,
        tacotron2_model_type=None,
        glow_tts_encoder=None,
        forward_attention=False,
        location_attention=True
    ):
        dataset, audio = data_loader(
            name = self.dataset_name, path = self.data_path, stats=stats_path
        )
        if self.dataset_name == "ljpseech":
            if model_name == "tacotron2":
                if tacotron2_model_type == "double decoder consistency":
                    model_config = self.model.single_speaker_tacotron2_DDC(
                        audio, dataset, forward_attn=forward_attention, location_attn=location_attention
                    )
                elif tacotron2_model_type == "dynamic convolution attention":
                    model_config = self.model.single_speaker_tacotron2_DCA(
                        audio, dataset, forward_attn=forward_attention, location_attn=location_attention
                    )
                else:
                    model_config = self.model.single_speaker_tacotron2_base(
                        audio, dataset, forward_attn=forward_attention, location_attn=location_attention
                    )
            elif model_name == "glow tts":
                model_config = self.model.ljspeech_glow_tts(audio, dataset, encoder=glow_tts_encoder)
            elif model_name == "vits tts":
                model_config = self.model.ljspeech_vits_tts(audio, dataset)
        elif self.dataset_name == "sam" or "sam_accenture":
            if model_name == "tacotron2":
                if tacotron2_model_type == "double decoder consistency":
                    model_config = self.model.single_speaker_tacotron2_DDC(
                        audio,
                        dataset,
                        pla=0.5,
                        dla=0.5,
                        ga=0.0,
                        forward_attn=forward_attention,
                        location_attn=location_attention,
                    )
            elif tacotron2_model_type == "dynamic convolution attention":
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

    def multi_speaker_autotts(
        self,
        model_name,
        speaker_file,
        glowtts_encoder
    ):
        '''
        This is the auto tts trainer for multispeaker training, currently only suppports
        glow tts and vits tts training on vctk dataset.
        '''
        if self.dataset_name == "vctk":
            if model_name == "glow tts":
                model_config = self.model.ScGlowTts(audio, dataset, speaker_file, encoder=glowtts_encoder)
        args, config, output_path, _, c_logger, tb_logger = init_training(TrainingArgs(), model_config)
        trainer = Trainer(args, config, output_path, c_logger, tb_logger)
        return trainer


class VocoderAutoTrainer:
    """This is trainer for calling complete recipes based off public datasets.
    all configs are based off pretrained model configs or the model papers.

    usage:
            From TTS.auto_tts.complete_recipes import VocoderExamples
            trainer = VocoderExamples(data_path='DEFINE THIS', batch_size=32, learning_rate=0.001,
                      mixed_precision=False, output_path='DEFINE THIS', epochs=1000)
            model = trainer.ljspeechAutoTts("hifigan")
            model.fit()
    """

    def __init__(self, data_path, dataset_name, batch_size, output_path, mixed_precision, gen_lr, disc_lr, epochs):

        self.data_path = data_path
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.output_path = output_path
        self.mixed_precision = mixed_precision
        self.gen_lr = gen_lr
        self.disc_lr = disc_lr
        self.epochs = epochs
        self.model = VocoderModels(
            batch_size=self.batch_size,
            mixed_precision=self.mixed_precision,
            generator_learning_rate=self.gen_lr,
            discriminator_learning_rate=self.disc_lr,
            epochs=self.epochs,
        )

    def single_speaker_autotts(
        self,
        model_name
    ):
        '''
        This is the funtion for calling a single speaker vocoder model to train on 
        one of the public datasets. it currently only supports ljspeech!!!
        '''
        if self.dataset_name == "ljspeech":
            _, audio = data_loader(name="ljspeech", path=self.data_path, stats_path=None)
            if model_name == "hifigan":
            model_config = self.model.ljspeechHifiGan(audio, self.data_path)
            elif model_name == "wavegrad":
                model_config = self.model.ljspeechWaveGrad(audio, self.data_path)
            elif model_name == "univnet":
                model_config = self.model.ljspeechUnivnet(audio, self.data_path)  # ToDo: add the stats path to the config
            elif model_name == "multiband melgan":
                model_config = self.model.ljspeechMultiBandMelGan(audio, self.data_path)
        args, config, output_path, _, c_logger, tb_logger = init_training(TrainingArgs(), model_config)
        trainer = Trainer(args, config, output_path, c_logger, tb_logger)
        return trainer
