# Mozilla TTS Vocoders (Experimental)

We provide here different vocoder implementations which can be combined with our TTS models to enable "FASTER THAN REAL-TIME" end-to-end TTS stack.

Currently, there are implementations of the following models.

- Melgan
- MultiBand-Melgan
- GAN-TTS (Discriminator Only)

It is also very easy to adapt different vocoder models as we provide here a flexible and modular (but not too modular) framework.

## Training a model

You can see here an example (Soon)[Colab Notebook]() training MelGAN with LJSpeech dataset.

In order to train a new model, you need to collecto all your wav files under a common parent folder and give this path to `data_path` field in '''config.json'''

You need to define other relevant parameters in your ```config.json``` and then start traning with the following command from Mozilla TTS root path.

```CUDA_VISIBLE_DEVICES='1' python vocoder/train.py --config_path path/to/config.json```

Exampled config files can be found under `vocoder/configs/` folder.

You can continue a previous training by the following command.

```CUDA_VISIBLE_DEVICES='1' python vocoder/train.py --continue_path path/to/your/model/folder```

You can fine-tune a pre-trained model by the following command.

```CUDA_VISIBLE_DEVICES='1' python vocoder/train.py --restore_path path/to/your/model.pth.tar```

Restoring a model starts a new training in a different output folder. It only restores model weights with the given checkpoint file. However, continuing a training starts from the same conditions the previous training run left off.

You can also follow your training runs on Tensorboard as you do with our TTS models.

## Acknowledgement
Thanks to @kan-bayashi for his [repository](https://github.com/kan-bayashi/ParallelWaveGAN) being the start point of our work.