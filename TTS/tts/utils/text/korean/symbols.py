# coding: utf-8
'''
Defines the set of symbols used in text input to the model.
The default is a set of ASCII characters that works well for English or text that has been run
through Unidecode. For other data, you can modify _characters. See TRAINING_DATA.md for details.
'''
from jamo import h2j, j2h
from jamo.jamo import _jamo_char_to_hcj

from .korean import ALL_SYMBOLS, PAD, EOS

# For english
en_symbols = PAD+EOS+'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!\'(),-.:;? '  #<-For deployment(Because korean ALL_SYMBOLS follow this convention)

symbols = ALL_SYMBOLS # for korean

"""
초성과 종성은 같아보이지만, 다른 character이다.
'_~ᄀᄁᄂᄃᄄᄅᄆᄇᄈᄉᄊᄋᄌᄍᄎᄏᄐᄑ하ᅢᅣᅤᅥᅦᅧᅨᅩᅪᅫᅬᅭᅮᅯᅰᅱᅲᅳᅴᅵᆨᆩᆪᆫᆬᆭᆮᆯᆰᆱᆲᆳᆴᆵᆶᆷᆸᆹᆺᆻᆼᆽᆾᆿᇀᇁᇂ!'(),-.:;? '
'_': 0, '~': 1, 'ᄀ': 2, 'ᄁ': 3, 'ᄂ': 4, 'ᄃ': 5, 'ᄄ': 6, 'ᄅ': 7, 'ᄆ': 8, 'ᄇ': 9, 'ᄈ': 10, 
'ᄉ': 11, 'ᄊ': 12, 'ᄋ': 13, 'ᄌ': 14, 'ᄍ': 15, 'ᄎ': 16, 'ᄏ': 17, 'ᄐ': 18, 'ᄑ': 19, 'ᄒ': 20, 
'ᅡ': 21, 'ᅢ': 22, 'ᅣ': 23, 'ᅤ': 24, 'ᅥ': 25, 'ᅦ': 26, 'ᅧ': 27, 'ᅨ': 28, 'ᅩ': 29, 'ᅪ': 30, 
'ᅫ': 31, 'ᅬ': 32, 'ᅭ': 33, 'ᅮ': 34, 'ᅯ': 35, 'ᅰ': 36, 'ᅱ': 37, 'ᅲ': 38, 'ᅳ': 39, 'ᅴ': 40, 
'ᅵ': 41, 'ᆨ': 42, 'ᆩ': 43, 'ᆪ': 44, 'ᆫ': 45, 'ᆬ': 46, 'ᆭ': 47, 'ᆮ': 48, 'ᆯ': 49, 'ᆰ': 50, 
'ᆱ': 51, 'ᆲ': 52, 'ᆳ': 53, 'ᆴ': 54, 'ᆵ': 55, 'ᆶ': 56, 'ᆷ': 57, 'ᆸ': 58, 'ᆹ': 59, 'ᆺ': 60, 
'ᆻ': 61, 'ᆼ': 62, 'ᆽ': 63, 'ᆾ': 64, 'ᆿ': 65, 'ᇀ': 66, 'ᇁ': 67, 'ᇂ': 68, '!': 69, "'": 70, 
'(': 71, ')': 72, ',': 73, '-': 74, '.': 75, ':': 76, ';': 77, '?': 78, ' ': 79
"""