# Converting Torch to TF 2

Currently, 🐸TTS supports the vanilla Tacotron2 and MelGAN models in TF 2.It does not support advanced attention methods and other small tricks used by the Torch models. You can convert any Torch model trained after v0.0.2.

You can also export TF 2 models to TFLite for even faster inference.

## How to convert from Torch to TF 2.0
Make sure you installed Tensorflow v2.2. It is not installed by default by :frog: TTS.

All the TF related code stays under ```tf``` folder.

To convert a **compatible** Torch model, run the following command with the right arguments:

```bash
python TTS/bin/convert_tacotron2_torch_to_tf.py\
        --torch_model_path /path/to/torch/model.pth.tar \
        --config_path /path/to/model/config.json\
        --output_path /path/to/output/tf/model
```

This will create a TF model file. Notice that our model format is not compatible with the official TF checkpoints. We created our custom format to match Torch checkpoints we use. Therefore, use the ```load_checkpoint``` and ```save_checkpoint``` functions provided under ```TTS.tf.generic_utils```.
