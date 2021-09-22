(what_makes_a_good_dataset)=
# What makes a good TTS dataset

## What Makes a Good Dataset
* **Gaussian like distribution on clip and text lengths**. So plot the distribution of clip lengths and check if it covers enough short and long voice clips.
* **Mistake free**. Remove any wrong or broken files. Check annotations, compare transcript and audio length.
* **Noise free**. Background noise might lead your model to struggle, especially for a good alignment. Even if it learns the alignment, the final result is likely to be suboptimial.
* **Compatible tone and pitch among voice clips**. For instance, if you are using audiobook recordings for your project, it might have impersonations for different characters in the book. These differences between samples downgrade the model performance.
* **Good phoneme coverage**. Make sure that your dataset covers a good portion of the phonemes, di-phonemes, and in some languages tri-phonemes.
* **Naturalness of recordings**. For your model WISIAIL (What it sees is all it learns). Therefore, your dataset should accommodate all the attributes you want to hear from your model.

## Preprocessing Dataset
If you like to use a bespoken dataset, you might like to perform a couple of quality checks before training. 🐸TTS provides a couple of notebooks (CheckSpectrograms, AnalyzeDataset) to expedite this part for you.

* **AnalyzeDataset** is for checking dataset distribution in terms of the clip and transcript lengths. It is good to find outlier instances (too long, short text but long voice clip, etc.)and remove them before training. Keep in mind that we like to have a good balance between long and short clips to prevent any bias in training. If you have only short clips (1-3 secs), then your model might suffer for long sentences and if your instances are long, then it might not learn the alignment or might take too long to train the model.

* **CheckSpectrograms** is to measure the noise level of the clips and find good audio processing parameters. The noise level might be observed by checking spectrograms. If spectrograms look cluttered, especially in silent parts, this dataset might not be a good candidate for a TTS project. If your voice clips are too noisy in the background, it makes things harder for your model to learn the alignment, and the final result might be different than the voice you are given.
If the spectrograms look good, then the next step is to find a good set of audio processing parameters, defined in ```config.json```. In the notebook, you can compare different sets of parameters and see the resynthesis results in relation to the given ground-truth. Find the best parameters that give the best possible synthesis performance.

Another practical detail is the quantization level of the clips. If your dataset has a very high bit-rate, that might cause slow data-load time and consequently slow training. It is better to reduce the sample-rate of your dataset to around 16000-22050.