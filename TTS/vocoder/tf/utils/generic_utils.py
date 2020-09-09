import re
import importlib


def to_camel(text):
    text = text.capitalize()
    return re.sub(r'(?!^)_([a-zA-Z])', lambda m: m.group(1).upper(), text)


def setup_generator(c):
    print(" > Generator Model: {}".format(c.generator_model))
    MyModel = importlib.import_module('TTS.vocoder.tf.models.' +
                                      c.generator_model.lower())
    MyModel = getattr(MyModel, to_camel(c.generator_model))
    if c.generator_model in 'melgan_generator':
        model = MyModel(
            in_channels=c.audio['num_mels'],
            out_channels=1,
            proj_kernel=7,
            base_channels=512,
            upsample_factors=c.generator_model_params['upsample_factors'],
            res_kernel=3,
            num_res_blocks=c.generator_model_params['num_res_blocks'])
    if c.generator_model in 'melgan_fb_generator':
        pass
    if c.generator_model in 'multiband_melgan_generator':
        model = MyModel(
            in_channels=c.audio['num_mels'],
            out_channels=4,
            proj_kernel=7,
            base_channels=384,
            upsample_factors=c.generator_model_params['upsample_factors'],
            res_kernel=3,
            num_res_blocks=c.generator_model_params['num_res_blocks'])
    return model
