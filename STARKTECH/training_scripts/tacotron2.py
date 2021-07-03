from TTS.trainer import Trainer, TrainingArgs, TensorboardLogger, TrainerCallback, init_training, ConsoleLogger
from TTS.tts.configs.tacotron2_config import Tacotron2Config
from TTS.tts.configs.tacotron_config import TacotronConfig
from TTS.tts.models.tacotron2 import Tacotron2

def example(name=None):
    if name == "model":
        args = TrainingArgs()
        config = Tacotron2Config()
        model = Tacotron2(config)
        trainer = Trainer(args, config, output_path="/home/logan", model=model)
        trainer.fit()
    elif name == "config":
        config = Tacotron2Config(data_path="/home/erogol/nvme/gdrive/Datasets/LJSpeech-1.1/wavs/", output_path="/home/logan", )
        args, config, output_path, _, c_logger, tb_logger = init_training(TrainingArgs(), config)
        trainer = Trainer(args, config, output_path, c_logger, tb_logger)
        trainer.fit()
    else:
        print("please specify if you want to see a model example or a config example")

class Tacotron2:
    """
    This is a class to instantly initualize different
    Tacotron2 models including single speaker, multi-speaker, and ddc
    with both model options and config options,
    all you have to do is just call .fit
    """
    def __init__(self, data_path, batch_size, output_path):
        self.data_path = data_path
        self.batch_size = batch_size
        self.output_path = output_path
    
    def single_speaker_model(self):
        args = TrainingArgs()
        config = Tacotron2Config()
        model = Tacotron2(config)
        tb_logger = TensorboardLogger()
        trainer = Trainer(args, config, tb_logger, output_path=self.output_path)
    
    def single_speaker_model_ddc(self):
        args = TrainingArgs()
        config = Tacotron2Config()
        model = Tacotron2(config)
        tb_logger = TensorboardLogger()
        trainer = Trainer(args, config, tb_logger, output_path=self.output_path)
    
    def multi_speaker_model_vctk(self):
        args = TrainingArgs()
        config = Tacotron2Config(use_gst=True)
        model = Tacotron2(config)
        tb_logger = TensorboardLogger()
        trainer = Trainer(args, config, tb_logger, output_path=self.output_path)
    
    def multi_speaker_ddc(self):
        args = TrainingArgs()
        config = Tacotron2Config()
        model = Tacotron2(config)
        tb_logger = TensorboardLogger()
        trainer = Trainer(args, config, tb_logger, output_path=self.output_path)
    
    def multi_speaker_model_libri_tts(self):
        args = TrainingArgs()
        config = Tacotron2Config()
        model = Tacotron2(config)
        tb_logger = TensorboardLogger()
        trainer = Trainer(args, config, tb_logger, output_path=self.output_path)
    
    def fine_tune_model(self):
        args = TrainingArgs()
        config = Tacotron2Config()
        model = Tacotron2(config)
        tb_logger = TensorboardLogger()
        trainer = Trainer(args, config, tb_logger, output_path=self.output_path)
        
    def single_speaker_config(self):
        config = Tacotron2Config(data_path=self.data_path, output_path=self.output_path)
        args, config, output_path, _, c_logger, tb_logger = init_training(TrainingArgs(), config)
        trainer = Trainer(args, config, output_path, c_logger, tb_logger)
    
    