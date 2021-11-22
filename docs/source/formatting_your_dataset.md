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

We recommend the following format delimited by `||`. In the following example, `audio1`, `audio2` refer to files `audio1.wav`, `audio2.wav` etc.

```
# metadata.txt

audio1||This is my sentence.
audio2||This is maybe my sentence.
audio3||This is certainly my sentence.
audio4||Let this be your sentence.
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

What you get out of a `formatter` is a `List[List[]]` in the following format.

```
>>> formatter(metafile_path)
[["audio1.wav", "This is my sentence.", "MyDataset"],
["audio1.wav", "This is maybe a sentence.", "MyDataset"],
...
]
```

Each sub-list is parsed as ```["<filename>", "<transcription>", "<speaker_name">]```.
```<speaker_name>``` is the dataset name for single speaker datasets and it is mainly used
in the multi-speaker models to map the speaker of the each sample. But for now, we only focus on single speaker datasets.

The purpose of a `formatter` is to parse your metafile and load the audio file paths and transcriptions. Then, its output passes to a `Dataset` object. It computes features from the audio signals, calls text normalization routines, and converts raw text to
phonemes if needed.

See `TTS.tts.datasets.TTSDataset`, a generic `Dataset` implementation for the `tts` models.

See `TTS.vocoder.datasets.*`, for different `Dataset` implementations for the `vocoder` models.

See `TTS.utils.audio.AudioProcessor` that includes all the audio processing and feature extraction functions used in a
`Dataset` implementation. Feel free to add things as you need.passed
