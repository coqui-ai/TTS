from TTS.tts.utils.text.cleaners import *

print(transliteration_cleaners("i want to eat hamberger"))
print(transliteration_cleaners(("나는 햄버거가 먹고 싶다")))
print(convert_to_ascii("나는 햄버거가 먹고 싶다"))
print(convert_to_ascii("今なら許してあげる。だから潔く盗んだものを返して"))
print(korean_cleaners("나는 5개의 사실을 알게되었다."))
print(english_cleaners("i want to eat ham"))
print(english_cleaners("배고파"))
print([""] + list("일이삼사오육칠팔구"))