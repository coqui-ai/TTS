(formatting_your_dataset)=
# Formatting Your Dataset

For training a TTS model, you need a dataset with speech recordings and transcriptions. The speech must be divided into audio clips and each clip needs transcription.

If you have a single audio file and you need to split it into clips, there are different open-source tools for you. We recommend Audacity. It is an open-source and free audio editing software.

It is also important to use a lossless audio file format to prevent compression artifacts. We recommend using `wav` file format.

Let's assume you created the audio clips and their transcription. You can collect all your clips under a folder. Let's call this folder `wavs`.

```
/wavs
  | - audio1.wav
  | - audio2.wav
  | - audio3.wav
  ...
```

You can either create separate transcription files for each clip or create a text file that maps each audio clip to its transcription. In this file, each line must be delimitered by a special character separating the audio file name from the transcription. And make sure that the delimiter is not used in the transcription text.

We recommend the following format delimited by `|`. In the following example, `audio1`, `audio2` refer to files `audio1.wav`, `audio2.wav` etc.

```
# metadata.txt

audio1|This is my sentence.
audio2|This is maybe my sentence.
audio3|This is certainly my sentence.
audio4|Let this be your sentence.
...
```

In the end, we have the following folder structure
```
/MyTTSDataset
      |
      | -> metadata.txt
      | -> /wavs
              | -> audio1.wav
              | -> audio2.wav
              | ...
```

The format above is taken from widely-used the [LJSpeech](https://keithito.com/LJ-Speech-Dataset/) dataset. You can also download and see the dataset. 🐸TTS already provides tooling for the LJSpeech. if you use the same format, you can start training your models right away.

## Dataset Quality

Your dataset should have good coverage of the target language. It should cover the phonemic variety, exceptional sounds and syllables. This is extremely important for especially non-phonemic languages like English.

For more info about dataset qualities and properties check our [post](https://github.com/coqui-ai/TTS/wiki/What-makes-a-good-TTS-dataset).

## Using Your Dataset in 🐸TTS

After you collect and format your dataset, you need to check two things. Whether you need a `formatter` and a `text_cleaner`. The `formatter` loads the text file (created above) as a list and the `text_cleaner` performs a sequence of text normalization operations that converts the raw text into the spoken representation (e.g. converting numbers to text, acronyms, and symbols to the spoken format).

If you use a different dataset format then the LJSpeech or the other public datasets that 🐸TTS supports, then you need to write your own `formatter`.

If your dataset is in a new language or it needs special normalization steps, then you need a new `text_cleaner`.

What you get out of a `formatter` is a `List[Dict]` in the following format.

```
>>> formatter(metafile_path)
[
    {"audio_file":"audio1.wav", "text":"This is my sentence.", "speaker_name":"MyDataset", "language": "lang_code"},
    {"audio_file":"audio1.wav", "text":"This is maybe a sentence.", "speaker_name":"MyDataset", "language": "lang_code"},
    ...
]
```

Each sub-list is parsed as ```{"<filename>", "<transcription>", "<speaker_name">]```.
```<speaker_name>``` is the dataset name for single speaker datasets and it is mainly used
in the multi-speaker models to map the speaker of the each sample. But for now, we only focus on single speaker datasets.

The purpose of a `formatter` is to parse your manifest file and load the audio file paths and transcriptions.
Then, the output is passed to the `Dataset`. It computes features from the audio signals, calls text normalization routines, and converts raw text to
phonemes if needed.

## Loading your dataset

Load one of the dataset supported by 🐸TTS.

```python
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples


# dataset config for one of the pre-defined datasets
dataset_config = BaseDatasetConfig(
    formatter="vctk", meta_file_train="", language="en-us", path="dataset-path")
)

# load training samples
train_samples, eval_samples = load_tts_samples(dataset_config, eval_split=True)
```

Load a custom dataset with a custom formatter.

```python
from TTS.tts.datasets import load_tts_samples


# custom formatter implementation
def formatter(root_path, manifest_file, **kwargs):  # pylint: disable=unused-argument
    """Assumes each line as ```<filename>|<transcription>```
    """
    txt_file = os.path.join(root_path, manifest_file)
    items = []
    speaker_name = "my_speaker"
    with open(txt_file, "r", encoding="utf-8") as ttf:
        for line in ttf:
            cols = line.split("|")
            wav_file = os.path.join(root_path, "wavs", cols[0])
            text = cols[1]
            items.append({"text":text, "audio_file":wav_file, "speaker_name":speaker_name, "root_path": root_path})
    return items

# load training samples
train_samples, eval_samples = load_tts_samples(dataset_config, eval_split=True, formatter=formatter)
```

See `TTS.tts.datasets.TTSDataset`, a generic `Dataset` implementation for the `tts` models.

See `TTS.vocoder.datasets.*`, for different `Dataset` implementations for the `vocoder` models.

See `TTS.utils.audio.AudioProcessor` that includes all the audio processing and feature extraction functions used in a
`Dataset` implementation. Feel free to add things as you need.
