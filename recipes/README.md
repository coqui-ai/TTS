# 🐸💬 TTS Training Recipes

TTS recipes intended to host scripts running all the necessary steps to train a TTS model on a particular dataset.

For each dataset, you need to download the dataset once. Then you run the training for the model you want.

Run each script from the root TTS folder as follows.

```console
$ sh ./recipes/<dataset>/download_<dataset>.sh
$ python recipes/<dataset>/<model_name>/train.py
```

For some datasets you might need to resample the audio files. For example, VCTK dataset can be resampled to 22050Hz as follows.

```console
python TTS/bin/resample.py --input_dir recipes/vctk/VCTK/wav48_silence_trimmed --output_sr 22050 --output_dir recipes/vctk/VCTK/wav48_silence_trimmed --n_jobs 8 --file_ext flac
```

If you train a new model using TTS, feel free to share your training to expand the list of recipes.

You can also open a new discussion and share your progress with the 🐸 community.
