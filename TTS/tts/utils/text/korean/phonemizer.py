from jamo import hangul_to_jamo

from TTS.tts.utils.text.korean.korean import normalize

g2p = None


def korean_text_to_phonemes(text, character: str = "hangeul") -> str:
    """

    The input and output values look the same, but they are different in Unicode.

    example :

        input = '하늘' (Unicode : \ud558\ub298), (하 + 늘)
        output = '하늘' (Unicode :\u1112\u1161\u1102\u1173\u11af), (ᄒ + ᅡ + ᄂ + ᅳ + ᆯ)

    """
    global g2p  # pylint: disable=global-statement
    if g2p is None:
        from g2pkk import G2p

        g2p = G2p()

    if character == "english":
        from anyascii import anyascii

        text = normalize(text)
        text = g2p(text)
        text = anyascii(text)
        return text

    text = normalize(text)
    text = g2p(text)
    text = list(hangul_to_jamo(text))  # '하늘' --> ['ᄒ', 'ᅡ', 'ᄂ', 'ᅳ', 'ᆯ']
    return "".join(text)
