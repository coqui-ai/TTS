import json
import os
import re

import pypinyin
import torch
from num2words import num2words
from tokenizers import Tokenizer

from TTS.tts.layers.xtts.zh_num2words import TextNorm as zh_num2words

_whitespace_re = re.compile(r"\s+")

# List of (regular expression, replacement) pairs for abbreviations:
_abbreviations = {
    "en": [
        (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])
        for x in [
            ("mrs", "misess"),
            ("mr", "mister"),
            ("dr", "doctor"),
            ("st", "saint"),
            ("co", "company"),
            ("jr", "junior"),
            ("maj", "major"),
            ("gen", "general"),
            ("drs", "doctors"),
            ("rev", "reverend"),
            ("lt", "lieutenant"),
            ("hon", "honorable"),
            ("sgt", "sergeant"),
            ("capt", "captain"),
            ("esq", "esquire"),
            ("ltd", "limited"),
            ("col", "colonel"),
            ("ft", "fort"),
        ]
    ],
    "es": [
        (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])
        for x in [
            ("sra", "señora"),
            ("sr", "señor"),
            ("dr", "doctor"),
            ("dra", "doctora"),
            ("st", "santo"),
            ("co", "compañía"),
            ("jr", "junior"),
            ("ltd", "limitada"),
        ]
    ],
    "fr": [
        (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])
        for x in [
            ("mme", "madame"),
            ("mr", "monsieur"),
            ("dr", "docteur"),
            ("st", "saint"),
            ("co", "compagnie"),
            ("jr", "junior"),
            ("ltd", "limitée"),
        ]
    ],
    "de": [
        (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])
        for x in [
            ("fr", "frau"),
            ("dr", "doktor"),
            ("st", "sankt"),
            ("co", "firma"),
            ("jr", "junior"),
        ]
    ],
    "pt": [
        (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])
        for x in [
            ("sra", "senhora"),
            ("sr", "senhor"),
            ("dr", "doutor"),
            ("dra", "doutora"),
            ("st", "santo"),
            ("co", "companhia"),
            ("jr", "júnior"),
            ("ltd", "limitada"),
        ]
    ],
    "it": [
        (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])
        for x in [
            # ("sig.ra", "signora"),
            ("sig", "signore"),
            ("dr", "dottore"),
            ("st", "santo"),
            ("co", "compagnia"),
            ("jr", "junior"),
            ("ltd", "limitata"),
        ]
    ],
    "pl": [
        (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])
        for x in [
            ("p", "pani"),
            ("m", "pan"),
            ("dr", "doktor"),
            ("sw", "święty"),
            ("jr", "junior"),
        ]
    ],
    "ar": [
        (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])
        for x in [
            # There are not many common abbreviations in Arabic as in English.
        ]
    ],
    "zh-cn": [
        (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])
        for x in [
            # Chinese doesn't typically use abbreviations in the same way as Latin-based scripts.
        ]
    ],
    "cs": [
        (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])
        for x in [
            ("dr", "doktor"),  # doctor
            ("ing", "inženýr"),  # engineer
            ("p", "pan"),  # Could also map to pani for woman but no easy way to do it
            # Other abbreviations would be specialized and not as common.
        ]
    ],
    "ru": [
        (re.compile("\\b%s\\b" % x[0], re.IGNORECASE), x[1])
        for x in [
            ("г-жа", "госпожа"),  # Mrs.
            ("г-н", "господин"),  # Mr.
            ("д-р", "доктор"),  # doctor
            # Other abbreviations are less common or specialized.
        ]
    ],
    "nl": [
        (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])
        for x in [
            ("dhr", "de heer"),  # Mr.
            ("mevr", "mevrouw"),  # Mrs.
            ("dr", "dokter"),  # doctor
            ("jhr", "jonkheer"),  # young lord or nobleman
            # Dutch uses more abbreviations, but these are the most common ones.
        ]
    ],
    "tr": [
        (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])
        for x in [
            ("b", "bay"),  # Mr.
            ("byk", "büyük"),  # büyük
            ("dr", "doktor"),  # doctor
            # Add other Turkish abbreviations here if needed.
        ]
    ],
}


def expand_abbreviations_multilingual(text, lang="en"):
    for regex, replacement in _abbreviations[lang]:
        text = re.sub(regex, replacement, text)
    return text


_symbols_multilingual = {
    "en": [
        (re.compile(r"%s" % re.escape(x[0]), re.IGNORECASE), x[1])
        for x in [
            ("&", " and "),
            ("@", " at "),
            ("%", " percent "),
            ("#", " hash "),
            ("$", " dollar "),
            ("£", " pound "),
            ("°", " degree "),
        ]
    ],
    "es": [
        (re.compile(r"%s" % re.escape(x[0]), re.IGNORECASE), x[1])
        for x in [
            ("&", " y "),
            ("@", " arroba "),
            ("%", " por ciento "),
            ("#", " numeral "),
            ("$", " dolar "),
            ("£", " libra "),
            ("°", " grados "),
        ]
    ],
    "fr": [
        (re.compile(r"%s" % re.escape(x[0]), re.IGNORECASE), x[1])
        for x in [
            ("&", " et "),
            ("@", " arobase "),
            ("%", " pour cent "),
            ("#", " dièse "),
            ("$", " dollar "),
            ("£", " livre "),
            ("°", " degrés "),
        ]
    ],
    "de": [
        (re.compile(r"%s" % re.escape(x[0]), re.IGNORECASE), x[1])
        for x in [
            ("&", " und "),
            ("@", " at "),
            ("%", " prozent "),
            ("#", " raute "),
            ("$", " dollar "),
            ("£", " pfund "),
            ("°", " grad "),
        ]
    ],
    "pt": [
        (re.compile(r"%s" % re.escape(x[0]), re.IGNORECASE), x[1])
        for x in [
            ("&", " e "),
            ("@", " arroba "),
            ("%", " por cento "),
            ("#", " cardinal "),
            ("$", " dólar "),
            ("£", " libra "),
            ("°", " graus "),
        ]
    ],
    "it": [
        (re.compile(r"%s" % re.escape(x[0]), re.IGNORECASE), x[1])
        for x in [
            ("&", " e "),
            ("@", " chiocciola "),
            ("%", " per cento "),
            ("#", " cancelletto "),
            ("$", " dollaro "),
            ("£", " sterlina "),
            ("°", " gradi "),
        ]
    ],
    "pl": [
        (re.compile(r"%s" % re.escape(x[0]), re.IGNORECASE), x[1])
        for x in [
            ("&", " i "),
            ("@", " małpa "),
            ("%", " procent "),
            ("#", " krzyżyk "),
            ("$", " dolar "),
            ("£", " funt "),
            ("°", " stopnie "),
        ]
    ],
    "ar": [
        # Arabic
        (re.compile(r"%s" % re.escape(x[0]), re.IGNORECASE), x[1])
        for x in [
            ("&", " و "),
            ("@", " على "),
            ("%", " في المئة "),
            ("#", " رقم "),
            ("$", " دولار "),
            ("£", " جنيه "),
            ("°", " درجة "),
        ]
    ],
    "zh-cn": [
        # Chinese
        (re.compile(r"%s" % re.escape(x[0]), re.IGNORECASE), x[1])
        for x in [
            ("&", " 和 "),
            ("@", " 在 "),
            ("%", " 百分之 "),
            ("#", " 号 "),
            ("$", " 美元 "),
            ("£", " 英镑 "),
            ("°", " 度 "),
        ]
    ],
    "cs": [
        # Czech
        (re.compile(r"%s" % re.escape(x[0]), re.IGNORECASE), x[1])
        for x in [
            ("&", " a "),
            ("@", " na "),
            ("%", " procento "),
            ("#", " křížek "),
            ("$", " dolar "),
            ("£", " libra "),
            ("°", " stupně "),
        ]
    ],
    "ru": [
        # Russian
        (re.compile(r"%s" % re.escape(x[0]), re.IGNORECASE), x[1])
        for x in [
            ("&", " и "),
            ("@", " собака "),
            ("%", " процентов "),
            ("#", " номер "),
            ("$", " доллар "),
            ("£", " фунт "),
            ("°", " градус "),
        ]
    ],
    "nl": [
        # Dutch
        (re.compile(r"%s" % re.escape(x[0]), re.IGNORECASE), x[1])
        for x in [
            ("&", " en "),
            ("@", " bij "),
            ("%", " procent "),
            ("#", " hekje "),
            ("$", " dollar "),
            ("£", " pond "),
            ("°", " graden "),
        ]
    ],
    "tr": [
        (re.compile(r"%s" % re.escape(x[0]), re.IGNORECASE), x[1])
        for x in [
            ("&", " ve "),
            ("@", " at "),
            ("%", " yüzde "),
            ("#", " diyez "),
            ("$", " dolar "),
            ("£", " sterlin "),
            ("°", " derece "),
        ]
    ],
}


def expand_symbols_multilingual(text, lang="en"):
    for regex, replacement in _symbols_multilingual[lang]:
        text = re.sub(regex, replacement, text)
        text = text.replace("  ", " ")  # Ensure there are no double spaces
    return text.strip()


_ordinal_re = {
    "en": re.compile(r"([0-9]+)(st|nd|rd|th)"),
    "es": re.compile(r"([0-9]+)(º|ª|er|o|a|os|as)"),
    "fr": re.compile(r"([0-9]+)(º|ª|er|re|e|ème)"),
    "de": re.compile(r"([0-9]+)(st|nd|rd|th|º|ª|\.(?=\s|$))"),
    "pt": re.compile(r"([0-9]+)(º|ª|o|a|os|as)"),
    "it": re.compile(r"([0-9]+)(º|°|ª|o|a|i|e)"),
    "pl": re.compile(r"([0-9]+)(º|ª|st|nd|rd|th)"),
    "ar": re.compile(r"([0-9]+)(ون|ين|ث|ر|ى)"),
    "cs": re.compile(r"([0-9]+)\.(?=\s|$)"),  # In Czech, a dot is often used after the number to indicate ordinals.
    "ru": re.compile(r"([0-9]+)(-й|-я|-е|-ое|-ье|-го)"),
    "nl": re.compile(r"([0-9]+)(de|ste|e)"),
    "tr": re.compile(r"([0-9]+)(\.|inci|nci|uncu|üncü|\.)"),
}
_number_re = re.compile(r"[0-9]+")
_currency_re = {
    "USD": re.compile(r"((\$[0-9\.\,]*[0-9]+)|([0-9\.\,]*[0-9]+\$))"),
    "GBP": re.compile(r"((£[0-9\.\,]*[0-9]+)|([0-9\.\,]*[0-9]+£))"),
    "EUR": re.compile(r"(([0-9\.\,]*[0-9]+€)|((€[0-9\.\,]*[0-9]+)))"),
}

_comma_number_re = re.compile(r"\b\d{1,3}(,\d{3})*(\.\d+)?\b")
_dot_number_re = re.compile(r"\b\d{1,3}(.\d{3})*(\,\d+)?\b")
_decimal_number_re = re.compile(r"([0-9]+[.,][0-9]+)")


def _remove_commas(m):
    text = m.group(0)
    if "," in text:
        text = text.replace(",", "")
    return text


def _remove_dots(m):
    text = m.group(0)
    if "." in text:
        text = text.replace(".", "")
    return text


def _expand_decimal_point(m, lang="en"):
    amount = m.group(1).replace(",", ".")
    return num2words(float(amount), lang=lang if lang != "cs" else "cz")


def _expand_currency(m, lang="en", currency="USD"):
    amount = float((re.sub(r"[^\d.]", "", m.group(0).replace(",", "."))))
    full_amount = num2words(amount, to="currency", currency=currency, lang=lang if lang != "cs" else "cz")

    and_equivalents = {
        "en": ", ",
        "es": " con ",
        "fr": " et ",
        "de": " und ",
        "pt": " e ",
        "it": " e ",
        "pl": ", ",
        "cs": ", ",
        "ru": ", ",
        "nl": ", ",
        "ar": ", ",
        "tr": ", ",
    }

    if amount.is_integer():
        last_and = full_amount.rfind(and_equivalents[lang])
        if last_and != -1:
            full_amount = full_amount[:last_and]

    return full_amount


def _expand_ordinal(m, lang="en"):
    return num2words(int(m.group(1)), ordinal=True, lang=lang if lang != "cs" else "cz")


def _expand_number(m, lang="en"):
    return num2words(int(m.group(0)), lang=lang if lang != "cs" else "cz")


def expand_numbers_multilingual(text, lang="en"):
    if lang == "zh-cn":
        text = zh_num2words()(text)
    else:
        if lang in ["en", "ru"]:
            text = re.sub(_comma_number_re, _remove_commas, text)
        else:
            text = re.sub(_dot_number_re, _remove_dots, text)
        try:
            text = re.sub(_currency_re["GBP"], lambda m: _expand_currency(m, lang, "GBP"), text)
            text = re.sub(_currency_re["USD"], lambda m: _expand_currency(m, lang, "USD"), text)
            text = re.sub(_currency_re["EUR"], lambda m: _expand_currency(m, lang, "EUR"), text)
        except:
            pass
        if lang != "tr":
            text = re.sub(_decimal_number_re, lambda m: _expand_decimal_point(m, lang), text)
        text = re.sub(_ordinal_re[lang], lambda m: _expand_ordinal(m, lang), text)
        text = re.sub(_number_re, lambda m: _expand_number(m, lang), text)
    return text


def lowercase(text):
    return text.lower()


def collapse_whitespace(text):
    return re.sub(_whitespace_re, " ", text)


def multilingual_cleaners(text, lang):
    text = text.replace('"', "")
    if lang == "tr":
        text = text.replace("İ", "i")
        text = text.replace("Ö", "ö")
        text = text.replace("Ü", "ü")
    text = lowercase(text)
    text = expand_numbers_multilingual(text, lang)
    text = expand_abbreviations_multilingual(text, lang)
    text = expand_symbols_multilingual(text, lang=lang)
    text = collapse_whitespace(text)
    return text


def basic_cleaners(text):
    """Basic pipeline that lowercases and collapses whitespace without transliteration."""
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text


def chinese_transliterate(text):
    return "".join(
        p[0] for p in pypinyin.pinyin(text, style=pypinyin.Style.TONE3, heteronym=False, neutral_tone_with_five=True)
    )


def japanese_cleaners(text, katsu):
    text = katsu.romaji(text)
    text = lowercase(text)
    return text


class VoiceBpeTokenizer:
    def __init__(self, vocab_file=None, preprocess=None):
        self.tokenizer = None
        self.katsu = None

        if vocab_file is not None:
            with open(vocab_file, "r", encoding="utf-8") as f:
                vocab = json.load(f)

            self.language = vocab["model"]["language"] if "language" in vocab["model"] else None

            if preprocess is None:
                self.preprocess = "pre_tokenizer" in vocab and vocab["pre_tokenizer"]
            else:
                self.preprocess = preprocess

            self.tokenizer = Tokenizer.from_file(vocab_file)

    def preprocess_text(self, txt, lang):
        if lang in ["en", "es", "fr", "de", "pt", "it", "pl", "ar", "cs", "ru", "nl", "tr", "zh-cn"]:
            txt = multilingual_cleaners(txt, lang)
            if lang == "zh-cn":
                txt = chinese_transliterate(txt)
        elif lang == "ja":
            if self.katsu is None:
                import cutlet

                self.katsu = cutlet.Cutlet()
            txt = japanese_cleaners(txt, self.katsu)
        else:
            raise NotImplementedError()
        return txt

    def encode(self, txt, lang):
        if self.preprocess:
            txt = self.preprocess_text(txt, lang)
        txt = f"[{lang}]{txt}"
        txt = txt.replace(" ", "[SPACE]")
        return self.tokenizer.encode(txt).ids

    def decode(self, seq):
        if isinstance(seq, torch.Tensor):
            seq = seq.cpu().numpy()
        txt = self.tokenizer.decode(seq, skip_special_tokens=False).replace(" ", "")
        txt = txt.replace("[SPACE]", " ")
        txt = txt.replace("[STOP]", "")
        txt = txt.replace("[UNK]", "")
        return txt

    def __len__(self):
        return self.tokenizer.get_vocab_size()

    def get_number_tokens(self):
        return max(self.tokenizer.get_vocab().values()) + 1
