import torch
import torchaudio
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC


def max_alignment(s1, s2, skip_character="~", record=None):
    """
    A clever function that aligns s1 to s2 as best it can. Wherever a character from s1 is not found in s2, a '~' is
    used to replace that character.

    Finally got to use my DP skills!
    """
    if record is None:
        record = {}
    assert skip_character not in s1, f"Found the skip character {skip_character} in the provided string, {s1}"
    if len(s1) == 0:
        return ""
    if len(s2) == 0:
        return skip_character * len(s1)
    if s1 == s2:
        return s1
    if s1[0] == s2[0]:
        return s1[0] + max_alignment(s1[1:], s2[1:], skip_character, record)

    take_s1_key = (len(s1), len(s2) - 1)
    if take_s1_key in record:
        take_s1, take_s1_score = record[take_s1_key]
    else:
        take_s1 = max_alignment(s1, s2[1:], skip_character, record)
        take_s1_score = len(take_s1.replace(skip_character, ""))
        record[take_s1_key] = (take_s1, take_s1_score)

    take_s2_key = (len(s1) - 1, len(s2))
    if take_s2_key in record:
        take_s2, take_s2_score = record[take_s2_key]
    else:
        take_s2 = max_alignment(s1[1:], s2, skip_character, record)
        take_s2_score = len(take_s2.replace(skip_character, ""))
        record[take_s2_key] = (take_s2, take_s2_score)

    return take_s1 if take_s1_score > take_s2_score else skip_character + take_s2


class Wav2VecAlignment:
    """
    Uses wav2vec2 to perform audio<->text alignment.
    """

    def __init__(self, device="cuda"):
        self.model = Wav2Vec2ForCTC.from_pretrained("jbetker/wav2vec2-large-robust-ft-libritts-voxpopuli").cpu()
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-large-960h")
        self.tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("jbetker/tacotron-symbols")
        self.device = device

    def align(self, audio, expected_text, audio_sample_rate=24000):
        orig_len = audio.shape[-1]

        with torch.no_grad():
            self.model = self.model.to(self.device)
            audio = audio.to(self.device)
            audio = torchaudio.functional.resample(audio, audio_sample_rate, 16000)
            clip_norm = (audio - audio.mean()) / torch.sqrt(audio.var() + 1e-7)
            logits = self.model(clip_norm).logits
            self.model = self.model.cpu()

        logits = logits[0]
        pred_string = self.tokenizer.decode(logits.argmax(-1).tolist())

        fixed_expectation = max_alignment(expected_text.lower(), pred_string)
        w2v_compression = orig_len // logits.shape[0]
        expected_tokens = self.tokenizer.encode(fixed_expectation)
        expected_chars = list(fixed_expectation)
        if len(expected_tokens) == 1:
            return [0]  # The alignment is simple; there is only one token.
        expected_tokens.pop(0)  # The first token is a given.
        expected_chars.pop(0)

        alignments = [0]

        def pop_till_you_win():
            if len(expected_tokens) == 0:
                return None
            popped = expected_tokens.pop(0)
            popped_char = expected_chars.pop(0)
            while popped_char == "~":
                alignments.append(-1)
                if len(expected_tokens) == 0:
                    return None
                popped = expected_tokens.pop(0)
                popped_char = expected_chars.pop(0)
            return popped

        next_expected_token = pop_till_you_win()
        for i, logit in enumerate(logits):
            top = logit.argmax()
            if next_expected_token == top:
                alignments.append(i * w2v_compression)
                if len(expected_tokens) > 0:
                    next_expected_token = pop_till_you_win()
                else:
                    break

        pop_till_you_win()
        if not (len(expected_tokens) == 0 and len(alignments) == len(expected_text)):
            torch.save([audio, expected_text], "alignment_debug.pth")
            assert False, (
                "Something went wrong with the alignment algorithm. I've dumped a file, 'alignment_debug.pth' to"
                "your current working directory. Please report this along with the file so it can get fixed."
            )

        # Now fix up alignments. Anything with -1 should be interpolated.
        alignments.append(orig_len)  # This'll get removed but makes the algorithm below more readable.
        for i in range(len(alignments)):
            if alignments[i] == -1:
                for j in range(i + 1, len(alignments)):
                    if alignments[j] != -1:
                        next_found_token = j
                        break
                for j in range(i, next_found_token):
                    gap = alignments[next_found_token] - alignments[i - 1]
                    alignments[j] = (j - i + 1) * gap // (next_found_token - i + 1) + alignments[i - 1]

        return alignments[:-1]

    def redact(self, audio, expected_text, audio_sample_rate=24000):
        if "[" not in expected_text:
            return audio
        splitted = expected_text.split("[")
        fully_split = [splitted[0]]
        for spl in splitted[1:]:
            assert "]" in spl, 'Every "[" character must be paired with a "]" with no nesting.'
            fully_split.extend(spl.split("]"))

        # At this point, fully_split is a list of strings, with every other string being something that should be redacted.
        non_redacted_intervals = []
        last_point = 0
        for i in range(len(fully_split)):
            if i % 2 == 0:
                end_interval = max(0, last_point + len(fully_split[i]) - 1)
                non_redacted_intervals.append((last_point, end_interval))
            last_point += len(fully_split[i])

        bare_text = "".join(fully_split)
        alignments = self.align(audio, bare_text, audio_sample_rate)

        output_audio = []
        for nri in non_redacted_intervals:
            start, stop = nri
            output_audio.append(audio[:, alignments[start] : alignments[stop]])
        return torch.cat(output_audio, dim=-1)
