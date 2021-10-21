import requests
import tqdm
import zipfile

from TTS.auto_tts.model_hub import TtsModels, VocoderModels
from TTS.auto_tts.utils import data_loader
from TTS.trainer import Trainer, TrainingArgs, init_training


class TtsAutoTrainer(TtsModels):
    """
    Args:

        data_path (str):
            The path to the dataset. Defaults to None.

        dataset (str):
            The dataset identifier. ex: ljspeech would be "ljspeech". Defaults to None.
            See auto_tts utils for specific dataset names.

        batch_size (int):
            The size the batches you pass to the model. This will depend on gpu memory.
            less than 32 is not recommended. Defaults to 32.

        output_path (str):
            The path where you want to the model config and model weights. If it is None it will
            use your current directory. Defaults to None

        mixed_precision (bool):
            enables mixed precision training. can make batch sizes bigger and make training faster.
            Could also make some trainings unstable. Defualts to False.

        learning_rate (float):
            The learning rate for the model. Defaults to 1e-3.

        epochs (int):
            how many times you want to model to go through the entire dataset. This usually doesn't need changing.
            Defaults to 1000.

    Usage:
        Python:
            From TTS.auto_tts.complete_recipes import TtsAutoTrainer
            trainer = TtsAutoTrainer(data_path='DEFINE THIS', stats_path=None, dataset="DEFINE THIS" batch_size=32, learning_rate=0.001,
                      mixed_precision=False, output_path='DEFINE THIS', epochs=1000)
            model = trainer.single_speaker_autotts("tacotron2, "double decoder consistency")
            model.fit()

        command line:
            python single_speaker_autotts.py --data_path ../LJSpeech-1.1 --dataset ljspeech --batch_size 32 --mixed_precision
            --model tacotron2 --tacotron2_model_type double decoder consistency --forward_attention
            --location_attention

    """

    def __init__(
        self,
        data_path=None,
        dataset=None,
        batch_size=32,
        output_path=None,
        mixed_precision=False,
        learning_rate=1e-3,
        epochs=1000,
    ):

        super().__init__(batch_size, mixed_precision, learning_rate, epochs, output_path)
        self.data_path = data_path
        self.dataset_name = dataset

    def single_speaker_autotts(     # im actually going to change this to autotts_recipes and i'm making a more generic
                                    # single_speaker_autotts cause it's gonna get too clunky when implenting fine tuning
                                    # all in the same function. it'll be finished in the next commit
        self,
        model_name,
        stats_path=None,
        tacotron2_model_type=None,
        glow_tts_encoder=None,
        forward_attention=False,
        location_attention=True,
        pretrained=False,
    ):
        """

        Args:
            model_name (str):
                name of the model you want to train. Defaults to None.


            stats_path (str):
                Optional, Stats path for the audio config if the model uses it. Defaults to None.


            tacotron2_model_type (str):
                Optional, Type of tacotron2 model you want to train, either double deocder consistency,
                or dynamic convolution attention. Defaults to None.


            glow_tts_encoder (str):
                 Optional, Type of encoder to train glow tts with. either transformer, gated,
                residual_bn, or time_depth. Defaults to None.


            forward_attention:
                Optional, Whether to use forward attention or not on tacotron2 models,
                Usaully makes the model allign faster. Defaults to False.


            location_attention:
                Optional, Whether to use location attention or not on Tacotron2 models. Defaults to True.


            pretrained (str):
                whether to use a pre trained model or not, This is recommended if you are training on
                custom data. Defaults to False

        """

        audio, dataset = data_loader(name=self.dataset_name, path=self.data_path, stats_path=stats_path)
        if self.dataset_name == "ljspeech":
            if model_name == "tacotron2":
                if tacotron2_model_type == "double decoder consistency":
                    model_config = self._single_speaker_tacotron2_DDC(
                        audio, dataset, forward_attn=forward_attention, location_attn=location_attention
                    )
                elif tacotron2_model_type == "dynamic convolution attention":
                    model_config = self._single_speaker_tacotron2_DCA(
                        audio, dataset, forward_attn=forward_attention, location_attn=location_attention
                    )
                else:
                    model_config = self._single_speaker_tacotron2_base(
                        audio, dataset, forward_attn=forward_attention, location_attn=location_attention
                    )
            elif model_name == "glow tts":
                model_config = self._single_speaker_glow_tts(audio, dataset, encoder=glow_tts_encoder)
            elif model_name == "vits tts":
                model_config = self._single_speaker_vits_tts(audio, dataset)
            elif model_name == "fast pitch":
                model_config = self._ljspeech_fast_fastpitch(audio, dataset)
        elif self.dataset_name == "baker":
            if model_name == "tacotron2":
                if tacotron2_model_type == "double decoder consistency":
                    model_config = self._single_speaker_tacotron2_DDC(
                        audio,
                        dataset,
                        pla=0.5,
                        dla=0.5,
                        ga=0.0,
                        forward_attn=forward_attention,
                        location_attn=location_attention,
                    )
                elif tacotron2_model_type == "dynamic convolution attention":
                    model_config = self._single_speaker_tacotron2_DCA(
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

    def multi_speaker_autotts(self, model_name, speaker_file, glowtts_encoder=None, r=2, forward_attn=True,
                              location_attn=False):
        """

        Args:
            location_attn (bool):
                enable location attention for tacotron2 model. Defaults to True.


            r (int):
                set the r for tacotron2 model. Defaults to 2.


            forward_attn (bool):
                set forward attention for tacotron2 model. Defaults to True.


            model_name (str):
                name of the model you want to train. Defaults to None.


            speaker_file (str):
                Path to either the d_vector file for glow_tts or speaker ids file for vits.
                Defaults to None


            glowtts_encoder:
                Optional, which encoder you want the glow tts model to use. Defaults to None.

        """
        audio, dataset = data_loader(name=self.dataset_name, path=self.data_path, stats_path=None)
        if self.dataset_name == "vctk":
            if model_name == "glow tts":
                model_config = self._sc_glow_tts(audio, dataset, speaker_file, encoder=glowtts_encoder)
            elif model_name == "vits tts":
                model_config = self._vctk_vits_tts(audio, dataset, speaker_file)
            elif model_name == "tacotron2":
                model_config = self._multi_speaker_vctk_tacotron2(audio, dataset, speaker_file, r=r, forward_attn=forward_attn,
                                                                  location_attn=location_attn)
        args, config, output_path, _, c_logger, tb_logger = init_training(TrainingArgs(), model_config)
        trainer = Trainer(args, config, output_path, c_logger, tb_logger)
        return trainer


class VocoderAutoTrainer(VocoderModels):
    """
    Args:

        data_path (str):
            The path to the dataset. Defaults to None.

        dataset (str):
            The dataset identifier. ex: ljspeech would be "ljspeech". Defaults to None.
            See auto_tts utils for specific dataset names.

        batch_size (int):
            The size the batches you pass to the model. This will depend on gpu memory.
            less than 32 is not recommended. Defaults to 32.

        output_path (str):
            The path where you want to the model config and model weights. If it is None it will
            use your current directory. Defaults to None

        mixed_precision (bool):
            enables mixed precision training. can make batch sizes bigger and make training faster.
            Could also make some trainings unstable. Defualts to False.

        learning_rate (List [float, float]):
            The learning rate for the model. This should be a list with the generator rate being first
            and discrimiator rate being second. Defaults to [1e-3, 1e-3].

        epochs (int):
            how many times you want to model to go through the entire dataset. This usually doesn't need changing.
            Defaults to 1000.

    Usage:
        Python:
            From TTS.auto_tts.complete_recipes import VocoderAutoTrainer
            trainer = VocoderAutoTrainer(data_path='DEFINE THIS', stats_path=None, dataset="DEFINE THIS",
                                         batch_size=32, learning_rate=[1e-3, 1e-3],
                                         mixed_precision=False, output_path='DEFINE THIS', epochs=1000)
            model = trainer.single_speaker_autotts("hifigan")
            model.fit()

        command line:
            python vocoder_autotts.py --data_path ../LJSpeech-1.1 --dataset ljspeech --batch_size 32 --mixed_precision
            --model hifigan

    """

    def __init__(
        self,
        data_path=None,
        dataset=None,
        batch_size=32,
        output_path=None,
        mixed_precision=False,
        learning_rate=None,
        epochs=1000,
    ):
        if learning_rate is None:
            learning_rate = [0.001, 0.001]
        super().__init__(
            batch_size,
            mixed_precision,
            generator_learning_rate=learning_rate[0],
            discriminator_learning_rate=learning_rate[1],
            epochs=epochs,
            output_path=output_path,
        )
        self.data_path: str = data_path
        self.dataset_name: str = dataset

    def single_speaker_autotts(self, model_name, stats_path=None):
        """
        Args:

            model_name (str):
                name of the model you want to train.

            Stats_path (str):
                Optional, Path to the stats file for the audio config. Defaults to None.

        """
        if self.dataset_name == "ljspeech":
            audio, _ = data_loader(name="ljspeech", path=self.data_path, stats_path=stats_path)
            if model_name == "hifigan":
                model_config = self._hifi_gan(audio, self.data_path)
            elif model_name == "wavegrad":
                model_config = self._wavegrad(audio, self.data_path)
            elif model_name == "univnet":
                model_config = self._univnet(audio, self.data_path)
            elif model_name == "multiband melgan":
                model_config = self._multiband_melgan(audio, self.data_path)
            elif model_name == "wavernn":
                model_config = self._wavernn(audio, self.data_path)
        args, config, output_path, _, c_logger, tb_logger = init_training(TrainingArgs(), model_config)
        trainer = Trainer(args, config, output_path, c_logger, tb_logger)
        return trainer

    def from_pretrained(model_name):
        pass
