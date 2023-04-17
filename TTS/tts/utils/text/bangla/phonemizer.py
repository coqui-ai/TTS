import re

import bangla
from bnnumerizer import numerize
from bnunicodenormalizer import Normalizer

# initialize
bnorm = Normalizer()


attribution_dict = {
    "সাঃ": "সাল্লাল্লাহু আলাইহি ওয়া সাল্লাম",
    "আঃ": "আলাইহিস সালাম",
    "রাঃ": "রাদিআল্লাহু আনহু",
    "রহঃ": "রহমাতুল্লাহি আলাইহি",
    "রহিঃ": "রহিমাহুল্লাহ",
    "হাফিঃ": "হাফিযাহুল্লাহ",
    "বায়ান": "বাইআন",
    "দাঃবাঃ": "দামাত বারাকাতুহুম,দামাত বারাকাতুল্লাহ",
    # "আয়াত" : "আইআত",#আইআত
    # "ওয়া" : "ওআ",
    # "ওয়াসাল্লাম"  : "ওআসাল্লাম",
    # "কেন"  : "কেনো",
    # "কোন" : "কোনো",
    # "বল"   : "বলো",
    # "চল"   : "চলো",
    # "কর"   : "করো",
    # "রাখ"   : "রাখো",
    "’": "",
    "‘": "",
    # "য়"     : "অ",
    # "সম্প্রদায়" : "সম্প্রদাই",
    # "রয়েছে"   : "রইছে",
    # "রয়েছ"    : "রইছ",
    "/": " বাই ",
}


def tag_text(text: str):
    # remove multiple spaces
    text = re.sub(" +", " ", text)
    # create start and end
    text = "start" + text + "end"
    # tag text
    parts = re.split("[\u0600-\u06FF]+", text)
    # remove non chars
    parts = [p for p in parts if p.strip()]
    # unique parts
    parts = set(parts)
    # tag the text
    for m in parts:
        if len(m.strip()) > 1:
            text = text.replace(m, f"{m}")
    # clean-tags
    text = text.replace("start", "")
    text = text.replace("end", "")
    return text


def normalize(sen):
    global bnorm  # pylint: disable=global-statement
    _words = [bnorm(word)["normalized"] for word in sen.split()]
    return " ".join([word for word in _words if word is not None])


def expand_full_attribution(text):
    for word, attr in attribution_dict.items():
        if word in text:
            text = text.replace(word, normalize(attr))
    return text


def collapse_whitespace(text):
    # Regular expression matching whitespace:
    _whitespace_re = re.compile(r"\s+")
    return re.sub(_whitespace_re, " ", text)


def bangla_text_to_phonemes(text: str) -> str:
    # english numbers to bangla conversion
    res = re.search("[0-9]", text)
    if res is not None:
        text = bangla.convert_english_digit_to_bangla_digit(text)

    # replace ':' in between two bangla numbers with ' এর '
    pattern = r"[০, ১, ২, ৩, ৪, ৫, ৬, ৭, ৮, ৯]:[০, ১, ২, ৩, ৪, ৫, ৬, ৭, ৮, ৯]"
    matches = re.findall(pattern, text)
    for m in matches:
        r = m.replace(":", " এর ")
        text = text.replace(m, r)

    # numerize text
    text = numerize(text)

    # tag sections
    text = tag_text(text)

    # text blocks
    # blocks = text.split("")
    # blocks = [b for b in blocks if b.strip()]

    # create tuple of (lang,text)
    if "" in text:
        text = text.replace("", "").replace("", "")
    # Split based on sentence ending Characters
    bn_text = text.strip()

    sentenceEnders = re.compile("[।!?]")
    sentences = sentenceEnders.split(str(bn_text))

    data = ""
    for sent in sentences:
        res = re.sub("\n", "", sent)
        res = normalize(res)
        # expand attributes
        res = expand_full_attribution(res)

        res = collapse_whitespace(res)
        res += "।"
        data += res
    return data
