<p align="center"><img src="https://user-images.githubusercontent.com/1402048/52643646-c2102980-2edd-11e9-8c37-b72f3c89a640.png" data-canonical-src="![TTS banner](https://user-images.githubusercontent.com/1402048/52643646-c2102980-2edd-11e9-8c37-b72f3c89a640.png =250x250)
" width="320" height="95" /></p>

<img src="https://travis-ci.org/mozilla/TTS.svg?branch=dev"/>

This project is a part of [Mozilla Common Voice](https://voice.mozilla.org/en). TTS aims a deep learning based Text2Speech engine, low in cost and high in quality. To begin with, you can hear a sample generated voice from [here](https://soundcloud.com/user-565970875/commonvoice-loc-sens-attn).

TTS includes two different model implementations which are based on [Tacotron](https://arxiv.org/abs/1703.10135) and [Tacotron2](https://arxiv.org/abs/1712.05884). Tacotron is smaller, efficient and easier to train but Tacotron2 provides better results, especially when it is combined with a Neural vocoder. Therefore, choose depending on your project requirements.

If you are new, you can also find [here](http://www.erogol.com/text-speech-deep-learning-architectures/) a brief post about TTS architectures and their comparisons.

## TTS Performance 
<p align="center"><img src="https://camo.githubusercontent.com/9fa79f977015e55eb9ec7aa32045555f60d093d3/68747470733a2f2f646973636f757273652d706161732d70726f64756374696f6e2d636f6e74656e742e73332e6475616c737461636b2e75732d656173742d312e616d617a6f6e6177732e636f6d2f6f7074696d697a65642f33582f362f342f363432386639383065396563373531633234386535393134363038393566373838316165633063365f325f363930783339342e706e67"/></p>

[Details...](https://github.com/mozilla/TTS/wiki/Mean-Opinion-Score-Results)

## Utilities under this Project
- Deep Learning based Text2Speech model.
- ```dataset_analysis```: Tools to curate a Text2Speech dataset.
- ```speaker_encoder```: Speaker Encoder model computing embedding vectors for voice files.
- ```server```: Basic server implementation with packaging. 

## Requirements and Installation
Highly recommended to use [miniconda](https://conda.io/miniconda.html) for easier installation.
  * python>=3.6
  * pytorch>=0.4.1
  * librosa
  * tensorboard
  * tensorboardX
  * matplotlib
  * unidecode

Install TTS using ```setup.py```. It will install all of the requirements automatically and make TTS available to all the python environment as an ordinary python module.

```python setup.py develop```

Or you can use ```requirements.txt``` to install the requirements only.

```pip install -r requirements.txt```

### Docker
A barebone `Dockerfile` exists at the root of the project, which should let you quickly setup the environment. By default, it will start the server and let you query it. Make sure to use `nvidia-docker` to use your GPUs. Make sure you follow the instructions in the [`server README`](server/README.md) before you build your image so that the server can find the model within the image.

```
docker build -t mozilla-tts .
nvidia-docker run -it --rm -p 5002:5002 mozilla-tts
```

## Checkpoints and Audio Samples
Please visit [our wiki.](https://github.com/mozilla/TTS/wiki/Released-Models)

## Example Model Outputs
Below you see Tacotron model state after 16K iterations with batch-size 32 with LJSpeech dataset.

> "Recent research at Harvard has shown meditating for as little as 8 weeks can actually increase the grey matter in the parts of the brain responsible for emotional regulation and learning."

Audio examples: [https://soundcloud.com/user-565970875](https://soundcloud.com/user-565970875)

<img src="images/example_model_output.png?raw=true" alt="example_output" width="400"/>

## Runtime
The most time-consuming part is the vocoder algorithm (Griffin-Lim) which runs on CPU. By setting its number of iterations lower, you might have faster execution with a small loss of quality. Some of the experimental values are below.

Sentence: "It took me quite a long time to develop a voice, and now that I have it I'm not going to be silent."

Audio length is approximately 6 secs.

| Time (secs) | System | # GL iters | Model
| ---- |:-------|:-----------| ---- |
|2.00|GTX1080Ti|30|Tacotron|
|3.01|GTX1080Ti|60|Tacotron|
|3.57|CPU|60|Tacotron|
|5.27|GTX1080Ti|60|Tacotron2|
|6.50|CPU|60|Tacotron2|


## Datasets and Data-Loading
TTS provides a generic dataloder easy to use for new datasets. You need to write an preprocessor function to integrate your own dataset.Check ```datasets/preprocess.py``` to see some examples. After the function, you need to set ```dataset``` field in ```config.json```. Do not forget other data related fields too.  

Some of the open-sourced datasets that we successfully applied TTS, are linked below.

- [LJ Speech](https://keithito.com/LJ-Speech-Dataset/)
- [Nancy](http://www.cstr.ed.ac.uk/projects/blizzard/2011/lessac_blizzard2011/)
- [TWEB](https://www.kaggle.com/bryanpark/the-world-english-bible-speech-dataset)
- [M-AI-Labs](http://www.caito.de/2019/01/the-m-ailabs-speech-dataset/)
- [LibriTTS](https://openslr.org/60/)
- [Spanish](https://drive.google.com/file/d/1Sm_zyBo67XHkiFhcRSQ4YaHPYM0slO_e/view?usp=sharing) - thx! @carlfm01

## Training and Fine-tuning LJ-Speech
Here you can find a [CoLab](https://gist.github.com/erogol/97516ad65b44dbddb8cd694953187c5b) notebook for a hands-on example, training LJSpeech. Or you can manually follow the guideline below. 

To start with, split ```metadata.csv``` into train and validation subsets respectively ```metadata_train.csv``` and ```metadata_val.csv```. Note that for text-to-speech, validation performance might be misleading since the loss value does not directly measure the voice quality to the human ear and it also does not measure the attention module performance. Therefore, running the model with new sentences and listening to the results is the best way to go. 

```
shuf metadata.csv > metadata_shuf.csv
head -n 12000 metadata_shuf.csv > metadata_train.csv
tail -n 1100 metadata_shuf.csv > metadata_val.csv
```

To train a new model, you need to define your own ```config.json``` file (check the example) and call with the command below. You also set the model architecture in  ```config.json```.

```train.py --config_path config.json```

To fine-tune a model, use ```--restore_path```.

```train.py --config_path config.json --restore_path /path/to/your/model.pth.tar```

For multi-GPU training use ```distribute.py```. It enables process based multi-GPU training where each process uses a single GPU.

```CUDA_VISIBLE_DEVICES="0,1,4" distribute.py --config_path config.json```

Each run creates a new output folder and ```config.json``` is copied under this folder.

In case of any error or intercepted execution, if there is no checkpoint yet under the output folder, the whole folder is going to be removed.

You can also enjoy Tensorboard,  if you point Tensorboard argument```--logdir``` to the experiment folder.

## Testing
Best way to test your network is to use Notebooks under ```notebooks``` folder.

There is also a good [CoLab](https://colab.research.google.com/github/tugstugi/dl-colab-notebooks/blob/master/notebooks/Mozilla_TTS_WaveRNN.ipynb) sample using pre-trained models (by @tugstugi).

## Contribution guidelines
This repository is governed by Mozilla's code of conduct and etiquette guidelines. For more details, please read the [Mozilla Community Participation Guidelines.](https://www.mozilla.org/about/governance/policies/participation/)

Please send your Pull Request to ```dev``` branch. Before making a Pull Request, check your changes for basic mistakes and style problems by using a linter. We have cardboardlinter setup in this repository, so for example, if you've made some changes and would like to run the linter on just the changed code, you can use the follow command:

```bash
pip install pylint cardboardlint
cardboardlinter --refspec master
```

## Collaborative Experimentation Guide
If you like to use TTS to try a new idea and like to share your experiments with the community, we urge you to use the following guideline for a better collaboration. 
(If you have an idea for better collaboration, let us know)
- Create a new branch.
- Open an issue pointing your branch. 
- Explain your experiment.
- Share your results as you proceed. (Tensorboard log files, audio results, visuals etc.)
- Use LJSpeech dataset (for English) if you like to compare results with the released models. (It is the most open scalable dataset for quick experimentation)

## Contact/Getting Help
- [Wiki](https://github.com/mozilla/TTS/wiki)

- [Discourse Forums](https://discourse.mozilla.org/c/tts) - If your question is not addressed in the Wiki, the Discourse Forums is the next place to look. They contain conversations on General Topics, Using TTS, and TTS Development.

- [Issues](https://github.com/mozilla/TTS/issues) - Finally, if all else fails, you can open an issue in our repo.

<!--## What is new with TTS
If you train TTS with LJSpeech dataset, you start to hear reasonable results after 12.5K iterations with batch size 32. This is the fastest training with character-based methods up to our knowledge. Out implementation is also quite robust against long sentences.
- Location sensitive attention ([ref](https://arxiv.org/pdf/1506.07503.pdf)). Attention is a vital part of text2speech models. Therefore, it is important to use an attention mechanism that suits the diagonal nature of the problem where the output strictly aligns with the text monotonically. Location sensitive attention performs better by looking into the previous alignment vectors and learns diagonal attention more easily. Yet, I believe there is a good space for research at this front to find a better solution.
- Attention smoothing with sigmoid ([ref](https://arxiv.org/pdf/1506.07503.pdf)). Attention weights are computed by normalized sigmoid values instead of softmax for sharper values. That enables the model to pick multiple highly scored inputs for alignments while reducing the noise.
- Weight decay ([ref](http://www.fast.ai/2018/07/02/adam-weight-decay/)). After a certain point of the training, you might observe the model over-fitting. That is, the model is able to pronounce words probably better but the quality of the speech quality gets lower and sometimes attention alignment gets disoriented.
- Stop token prediction with an additional module. The original Tacotron model does not propose a stop token to stop the decoding process. Therefore, you need to use heuristic measures to stop the decoder. Here, we prefer to use additional layers at the end to decide when to stop.
- Applying sigmoid to the model outputs. Since the output values are expected to be in the range [0, 1], we apply sigmoid to make things easier to approximate the expected output distribution.
- Phoneme based training is enabled for easier learning and robust pronunciation. It also makes easier to adapt TTS to the most languages without worrying about language specific characters.
- Configurable attention windowing at inference-time for robust alignment. It enforces network to only consider a certain window of encoder steps per iteration.
- Detailed Tensorboard stats for activation, weight and gradient values per layer. It is useful to detect defects and compare networks.
- Constant history window. Instead of using only the last frame of predictions, define a constant history queue. It enables training with gradually decreasing prediction frame (r=5 -> r=1) by only changing the last layer. For instance, you can train the model with r=5 and then fine-tune it with r=1 without any performance loss. It also solves well-known PreNet problem [#50](https://github.com/mozilla/TTS/issues/50). 
- Initialization of hidden decoder states with Embedding layers instead of zero initialization. 
One common question is to ask why we don't use Tacotron2 architecture. According to our ablation experiments, nothing, except Location Sensitive Attention, improves the performance, given the increase in the model size.
Please feel free to offer new changes and pull things off. We are happy to discuss and make things better.
-->

## Major TODOs
- [x] Implement the model.
- [x] Generate human-like speech on LJSpeech dataset.
- [x] Generate human-like speech on a different dataset (Nancy) (TWEB).
- [x] Train TTS with r=1 successfully.
- [x] Enable process based distributed training. Similar to (https://github.com/fastai/imagenet-fast/).
- [x] Adapting Neural Vocoder. TTS works with WaveRNN and ParallelWaveGAN (https://github.com/erogol/WaveRNN and https://github.com/erogol/ParallelWaveGAN)
- [ ] Multi-speaker embedding.
- [ ] Model optimization (model export, model pruning etc.)

<!--## References
- [Efficient Neural Audio Synthesis](https://arxiv.org/pdf/1802.08435.pdf)
- [Attention-Based models for speech recognition](https://arxiv.org/pdf/1506.07503.pdf)
- [Generating Sequences With Recurrent Neural Networks](https://arxiv.org/pdf/1308.0850.pdf)
- [Char2Wav: End-to-End Speech Synthesis](https://openreview.net/pdf?id=B1VWyySKx)
- [VoiceLoop: Voice Fitting and Synthesis via a Phonological Loop](https://arxiv.org/pdf/1707.06588.pdf)
- [WaveRNN](https://arxiv.org/pdf/1802.08435.pdf)
- [Faster WaveNet](https://arxiv.org/abs/1611.09482)
- [Parallel WaveNet](https://arxiv.org/abs/1711.10433)
-->

### References
- https://github.com/keithito/tacotron (Dataset pre-processing)
- https://github.com/r9y9/tacotron_pytorch (Initial Tacotron architecture)
