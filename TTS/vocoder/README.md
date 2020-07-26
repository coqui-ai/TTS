# Mozilla TTS Vocoders (Experimental)

Here there are vocoder model implementations which can be combined with the other TTS models.

Currently, following models are implemented:

- Melgan
- MultiBand-Melgan
- ParallelWaveGAN
- GAN-TTS (Discriminator Only)

It is also very easy to adapt different vocoder models as we provide a flexible and modular (but not too modular) framework.

## Training a model

You can see here an example (Soon)[Colab Notebook]() training MelGAN with LJSpeech dataset.

In order to train a new model, you need to gather all wav files into a folder and give this folder to `data_path` in '''config.json'''

You need to define other relevant parameters in your ```config.json``` and then start traning with the following command.

```CUDA_VISIBLE_DEVICES='0' python tts/bin/train_vocoder.py --config_path path/to/config.json```

Example config files can be found under `tts/vocoder/configs/` folder.

You can continue a previous training run by the following command.

```CUDA_VISIBLE_DEVICES='0' python tts/bin/train_vocoder.py --continue_path path/to/your/model/folder```

You can fine-tune a pre-trained model by the following command.

```CUDA_VISIBLE_DEVICES='0' python tts/bin/train_vocoder.py --restore_path path/to/your/model.pth.tar```

Restoring a model starts a new training in a different folder. It only restores model weights with the given checkpoint file. However, continuing a training starts from the same directory where the previous training run left off.

You can also follow your training runs on Tensorboard as you do with our TTS models.

## Acknowledgement
Thanks to @kan-bayashi for his [repository](https://github.com/kan-bayashi/ParallelWaveGAN) being the start point of our work.
