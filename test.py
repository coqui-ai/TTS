import torch
from TTS.tts.utils.text.cleaners import korean_cleaners, basic_cleaners
from typing import Callable, Dict, List, Union


from typing import Dict, List

from TTS.tts.utils.text.phonemizers import DEF_LANG_TO_PHONEMIZER, get_phonemizer_by_name
from TTS.tts.utils.text.phonemizers.multi_phonemizer import MultiPhonemizer

def e(text: str):
    for i in text:
        print(i)

if __name__ == "__main__":
     texts = {
         "en-us": "Hello, this is English example!",
         "zh-cn": "这是中国的例子",
         "ja-jp": "今なら許してあげる。だから潔く盗んだものを返して。",
         "ko-kr": "안녕하세요. 이 문장은 한국어 문장입니다."
     }
     phonemes = {}
     ph = MultiPhonemizer()
     for lang, text in texts.items():
         phoneme = ph.phonemize(text, lang)
         phonemes[lang] = phoneme
     print(phonemes)

     text = " as all books Not primarily intended as picture-books consist principally of types composed to form letterpress"
     text2 = "             는 배가 고파서 햄버거를 먹을 겁니다."
     print(basic_cleaners(text2))
