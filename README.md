# TTS
This project is a part of [Mozilla Common Voice](https://voice.mozilla.org/en). TTS aims a Text2Speech engine low in cost and high in quality. To begin with, you can hear a sample [here](https://soundcloud.com/user-565970875/commonvoice-loc-sens-attn).

The model here is highly inspired from Tacotron: [A Fully End-to-End Text-To-Speech Synthesis Model](https://arxiv.org/abs/1703.10135) however, it has many important updates over the baseline model that make training faster and computationally very efficient. Feel free to experiment new ideas and propose changes. 

You can find [here](http://www.erogol.com/text-speech-deep-learning-architectures/) a brief note pointing possible TTS architectures and their comparisons.

## Requirements and Installation
Highly recommended to use [miniconda](https://conda.io/miniconda.html) for easier installation.
  * python 3.6
  * pytorch 0.4
  * librosa
  * tensorboard
  * tensorboardX
  * matplotlib
  * unidecode

Install TTS using ```setup.py```. It will install all of the requirements automatically and make TTS available to all python environment as an ordinary python module. This makes things easier to run your model outside of the project folder.

```python setup.py develop```

Or you can use ```requirements.txt``` to install the requirements only.

```pip install -r requirements.txt```

## Checkpoints and Audio Samples
Checkout [here](https://mycroft.ai/blog/available-voices/#the-human-voice-is-the-most-perfect-instrument-of-all-arvo-part) to compare the samples (except the first) below.

| Models        | Commit            | Audio Sample  | Details |
| ------------- |:-----------------:|:--------------|:--------|
| [iter-62410](https://drive.google.com/open?id=1pjJNzENL3ZNps9n7k_ktGbpEl6YPIkcZ)| [99d56f7](https://github.com/mozilla/TTS/tree/99d56f7e93ccd7567beb0af8fcbd4d24c48e59e9)           | [link](https://soundcloud.com/user-565970875/99d56f7-iter62410 )|First model with plain Tacotron implementation.|
| [iter-170K](https://drive.google.com/open?id=16L6JbPXj6MSlNUxEStNn28GiSzi4fu1j) | [e00bc66](https://github.com/mozilla/TTS/tree/e00bc66) |[link](https://soundcloud.com/user-565970875/april-13-2018-07-06pm-e00bc66-iter170k)|More stable and longer trained model.|
| [iter-270K](https://drive.google.com/drive/folders/1Q6BKeEkZyxSGsocK2p_mqgzLwlNvbHFJ?usp=sharing)|[256ed63](https://github.com/mozilla/TTS/tree/256ed63)|[link](https://soundcloud.com/user-565970875/sets/samples-1650226)|Stop-Token prediction is added, to detect end of speech.|
| Best: [iter-120K](https://drive.google.com/open?id=1A5Hr6aSvfGgIiE20mBkpzyn3vvbR2APj) | [bf7590](https://github.com/mozilla/TTS/tree/bf7590) | [link](https://soundcloud.com/user-565970875/sets/september-26-2018-bf7590) | Better for longer sentences |

## Example Model Outputs
Below you see model state after 16K iterations with batch-size 32. 

> "Recent research at Harvard has shown meditating for as little as 8 weeks can actually increase the grey matter in the parts of the brain responsible for emotional regulation and learning."

Audio output: [https://soundcloud.com/user-565970875/iter16k-f48c3b](https://soundcloud.com/user-565970875/iter16k-f48c3b)

![example_model_output](images/example_model_output.png?raw=true)

## Runtime
The most time-consuming part is the vocoder algorithm (Griffin-Lim) which runs on CPU. By setting its number of iterations, you might have faster execution with a small loss of quality. Some of the experimental values are below.

Sentence: "It took me quite a long time to develop a voice, and now that I have it I'm not going to be silent."

Audio length is approximately 6 secs.

| Time (secs) | System | # GL iters |
| ---- |:-------|:-----------|
|2.00|GTX1080Ti|30|
|3.01|GTX1080Ti|60|


## Data
Currently TTS provides data loaders for datasets depicted below. It is also very is to adapt new datasets with few changes.
- [LJ Speech](https://keithito.com/LJ-Speech-Dataset/)

## Training and Fine-tuning
Split ```metadata.csv``` into train and validation subsets respectively ```metadata_train.csv``` and ```metadata_val.csv```. Note that having a validation split does not work well as oppose to other ML problems since at the validation time model generates spectrogram slices without "Teacher-Forcing" and that leads misalignment between the ground-truth and the prediction. Therefore, validation loss does not really show the model performance. Rather, you might use the all data for training and check the model performance by relying on human inspection.

```
shuf metadata.csv > metadata_shuf.csv
head -n 12000 metadata_shuf.csv > metadata_train.csv
tail -n 11000 metadata_shuf.csv > metadata_val.csv
```

To train a new model, you need to define a ```config.json``` file (simple template below) and call with the command below.

```train.py --config_path config.json```

To fine-tune a model, use ```--restore_path```.

```train.py --config_path config.json --restore_path /path/to/your/model.pth.tar```

If you like to use specific set of GPUs, you need set an environment variable. The code uses automatically all the available GPUs for data parallel training. If you don't specify the GPUs, it uses the all.

```CUDA_VISIBLE_DEVICES="0,1,4" train.py --config_path config.json```

Each run creates an experiment folder with some meta information, under the folder you set in ```config.json```. Also a copy of ```config.json``` is moved under the experiment folder for reproducibility. 

In case of any error or intercepted execution, if there is no checkpoint yet under the execution folder, the whole folder is going to be removed.

You can also enjoy Tensorboard, if you point the Tensorboard argument```--logdir``` to the experiment folder.

## Testing
Best way to test your pre-trained network is to use Notebooks under ```notebooks``` folder.

## What is new with TTS
If you train TTS with LJSpeech dataset, you start to hear reasonable results after 12.5K iterations with batch size 32. This is the fastest training with character based methods up to our knowledge. Out implementation is also quite robust against long sentences.

- Location sensitive attention ([ref](https://arxiv.org/pdf/1506.07503.pdf)). Attention is the vital part of text2speech models. Therefore, it is important to use an attention mechanism that suits the diagonal nature of the problem where the output strictly aligns with the text monotonically. Location sensitive attention performs better by looking into the previous alignment vectors and learns diagonal attention more easily. Yet, I believe there is a good space for research at this front to find a better solution.
- Attention smoothing with sigmoid ([ref](https://arxiv.org/pdf/1506.07503.pdf)). Attention weights are computed by normalized sigmoid values instead of softmax for sharper values. That enables the model to pick multiple highly scored inputs for alignments while reducing the noise. 
- Weight decay ([ref](http://www.fast.ai/2018/07/02/adam-weight-decay/)). After a certain point of the training, you might observe the model over-fitting. That is, model is able to pronounce words probably better but quality of the speech quality gets lower and sometimes attention alignment gets disoriented. 
- Stop token prediction with an additional module. The original Tacotron model does not propose a stop token to stop the decoding process. Therefore, you need to use heuristic measures to stop the decoder. Here, we prefer to use additional layers at the end to decide when to stop.
- Applying sigmoid to the model outputs. Since the output values are expected to be in the range [0, 1], we apply sigmoid to make things easier to approximate the expected output distribution. 

One common question is to ask why we don't use Tacotron2 architecture. According to our ablation experiments, nothing, except Location Sensitive Attention, improves the baseline perfomance of vanilla Tacotron. 
Please feel free to offer new changes and pull things off. We are happy to discuss and make things better. 

## Problems waiting to be solved, based on LJSpeech Dataset
- Punctuations at the end of a sentence affects the pronounciation of the last word in certain cases. Because punctuation sign is attended by the attention module , that forces network to create a voice signal or at least modify the voice signal being generated for neighboring frames. However, punctuation should only affect the general flow (like ? and !) or pauses in the sentence (. or ,).
- Simpler stop-token prediction. Right now we use RNN to keep the history of the previous frames. However, we never tested, if something simpler would work as well.
- Train for better mel-specs. Mel-spectrograms are not good enough to be fed to Neural Vocoder. Therefore, we needs better ways to improve the quality. These might be using adversarial training or some other trick ued by image generation methods. At the end, it is partially a text to image problem.
- irregular words: "minute", "focus", "aren't" etc. Even though, it might be solved by a larger or better dataset, some of irregular words cause network to mis-pronounce. Irregular means in this context is that written form and the pronounciation of the word have a unique difference.

## Major TODOs
- [x] Implement the model.
- [x] Generate human-like speech on LJSpeech dataset.
- [ ] Generate human-like speech on a different dataset.
- [ ] Adapting Neural Vocoder.
- [ ] Multi-speaker embedding.

## References
- [Efficient Neural Audio Synthesis](https://arxiv.org/pdf/1802.08435.pdf)
- [Attention-Based models for speech recognition](https://arxiv.org/pdf/1506.07503.pdf)
- [Generating Sequences With Recurrent Neural Networks](https://arxiv.org/pdf/1308.0850.pdf)
- [Char2Wav: End-to-End Speech Synthesis](https://openreview.net/pdf?id=B1VWyySKx)
- [VoiceLoop: Voice Fitting and Synthesis via a Phonological Loop](https://arxiv.org/pdf/1707.06588.pdf)
- [WaveRNN](https://arxiv.org/pdf/1802.08435.pdf)
- [Faster WaveNet](https://arxiv.org/abs/1611.09482)
- [Parallel WaveNet](https://arxiv.org/abs/1711.10433)

### Precursor implementations
- https://github.com/keithito/tacotron (Dataset and Test processing)
- https://github.com/r9y9/tacotron_pytorch (Initial Tacotron architecture)
