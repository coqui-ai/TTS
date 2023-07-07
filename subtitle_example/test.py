import sys
sys.path.append("..")
from TTS.api import TTS

tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DCA", progress_bar=True)
# tts = TTS(model_name="tts_models/en/vctk/vits", progress_bar=True)
# tts = TTS(model_name="tts_models/en/ljspeech/neural_hmm", progress_bar=True)

text_file = open("text_to_convert.txt", "r", encoding="utf-8")
text_to_convert = text_file.read()
text_file.close()

tts.tts_to_file(text=text_to_convert, speed=0.25, generate_subtitles=True)