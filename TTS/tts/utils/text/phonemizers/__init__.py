from TTS.tts.utils.text.phonemizers.bangla_phonemizer import BN_Phonemizer
from TTS.tts.utils.text.phonemizers.base import BasePhonemizer
from TTS.tts.utils.text.phonemizers.belarusian_phonemizer import BEL_Phonemizer
from TTS.tts.utils.text.phonemizers.espeak_wrapper import ESpeak
from TTS.tts.utils.text.phonemizers.gruut_wrapper import Gruut
from TTS.tts.utils.text.phonemizers.ko_kr_phonemizer import KO_KR_Phonemizer
from TTS.tts.utils.text.phonemizers.zh_cn_phonemizer import ZH_CN_Phonemizer

try:
    from TTS.tts.utils.text.phonemizers.ja_jp_phonemizer import JA_JP_Phonemizer
except ImportError:
    JA_JP_Phonemizer = None
    pass

PHONEMIZERS = {b.name(): b for b in (ESpeak, Gruut, KO_KR_Phonemizer, BN_Phonemizer)}


ESPEAK_LANGS = list(ESpeak.supported_languages().keys())
GRUUT_LANGS = list(Gruut.supported_languages())


# Dict setting default phonemizers for each language
# Add Gruut languages
_ = [Gruut.name()] * len(GRUUT_LANGS)
DEF_LANG_TO_PHONEMIZER = dict(list(zip(GRUUT_LANGS, _)))


# Add ESpeak languages and override any existing ones
_ = [ESpeak.name()] * len(ESPEAK_LANGS)
_new_dict = dict(list(zip(list(ESPEAK_LANGS), _)))
DEF_LANG_TO_PHONEMIZER.update(_new_dict)


# Force default for some languages
DEF_LANG_TO_PHONEMIZER["en"] = DEF_LANG_TO_PHONEMIZER["en-us"]
DEF_LANG_TO_PHONEMIZER["zh-cn"] = ZH_CN_Phonemizer.name()
DEF_LANG_TO_PHONEMIZER["ko-kr"] = KO_KR_Phonemizer.name()
DEF_LANG_TO_PHONEMIZER["bn"] = BN_Phonemizer.name()
DEF_LANG_TO_PHONEMIZER["be"] = BEL_Phonemizer.name()


# JA phonemizer has deal breaking dependencies like MeCab for some systems.
# So we only have it when we have it.
if JA_JP_Phonemizer is not None:
    PHONEMIZERS[JA_JP_Phonemizer.name()] = JA_JP_Phonemizer
    DEF_LANG_TO_PHONEMIZER["ja-jp"] = JA_JP_Phonemizer.name()


def get_phonemizer_by_name(name: str, **kwargs) -> BasePhonemizer:
    """Initiate a phonemizer by name

    Args:
        name (str):
            Name of the phonemizer that should match `phonemizer.name()`.

        kwargs (dict):
            Extra keyword arguments that should be passed to the phonemizer.
    """
    if name == "espeak":
        return ESpeak(**kwargs)
    if name == "gruut":
        return Gruut(**kwargs)
    if name == "zh_cn_phonemizer":
        return ZH_CN_Phonemizer(**kwargs)
    if name == "ja_jp_phonemizer":
        if JA_JP_Phonemizer is None:
            raise ValueError(" ❗ You need to install JA phonemizer dependencies. Try `pip install TTS[ja]`.")
        return JA_JP_Phonemizer(**kwargs)
    if name == "ko_kr_phonemizer":
        return KO_KR_Phonemizer(**kwargs)
    if name == "bn_phonemizer":
        return BN_Phonemizer(**kwargs)
    if name == "be_phonemizer":
        return BEL_Phonemizer(**kwargs)
    raise ValueError(f"Phonemizer {name} not found")


if __name__ == "__main__":
    print(DEF_LANG_TO_PHONEMIZER)
