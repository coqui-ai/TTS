from TTS.utils.synthesizer import Synthesizer
import mlflow

try:
    experiment = mlflow.get_experiment_by_name("tts: tts test")
    if experiment is None:
        experiment = mlflow.create_experiment("tts: tts test", artifact_location="s3://mlflow/TTS")
        experiment = mlflow.get_experiment_by_name("tts: tts test")
except mlflow.exceptions.RestException as e:
    experiment = mlflow.create_experiment("tts: tts test", artifact_location="s3://mlflow/TTS")
    experiment = mlflow.get_experiment_by_name("tts: tts test")


class MyModel(mlflow.pyfunc.PythonModel):
    '''
    def __init__(self, tts_path, tts_checkpoint):
        import os
        self.checkpoint = os.path.join(tts_path, tts_checkpoint)
        self.config_path = os.path.join(tts_path, "config.json")
        self.synthesizer = None

    def predict(self, context, model_input):
        self.synthesizer = Synthesizer(self.checkpoint, self.config_path)
        wav = self.synthesizer.tts(model_input)
        self.synthesizer.save_wav(wav, 'output.wav')

        return wav
    '''
    def __init__(self, synthesizer):
        self.synthesizer = synthesizer


    def predict(self, context, model_input):
        synthesizer = self.synthesizer
        wav = synthesizer.tts(model_input)
        synthesizer.save_wav(wav, 'output.wav')

        return wav