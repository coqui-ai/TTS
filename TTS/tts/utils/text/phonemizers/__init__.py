from TTS.tts.utils.text.phonemizers.base import BasePhonemizer
from TTS.tts.utils.text.phonemizers.espeak_wrapper import ESpeak
from TTS.tts.utils.text.phonemizers.gruut_wrapper import Gruut


PHONEMIZERS = {b.name(): b for b in (ESpeak, Gruut)}


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

    raise ValueError(f"Phonemizer {name} not found")


if __name__ == "__main__":
    print(DEF_LANG_TO_PHONEMIZER)
