### Speaker embedding (Experimental)

This is an implementation of https://arxiv.org/abs/1710.10467. This model can be used for voice and speaker embedding. So you can generate d-vectors for multi-speaker TTS or prune bad samples from your TTS dataset. Below is an example showing embedding results of various speakers. You can generate the same plot with the provided notebook. 

![](https://user-images.githubusercontent.com/1402048/64603079-7fa5c100-d3c8-11e9-88e7-88a00d0e37d1.png)

To run the code, you need to follow the same flow as in TTS. 

- Define 'config.json' for your needs. Note that, audio parameters should match your TTS model.
- Example training call ```python speaker_encoder/train.py --config_path speaker_encoder/config.json --data_path ~/Data/Libri-TTS/train-clean-360```
- Generate embedding vectors ```python speaker_encoder/compute_embeddings.py --use_cuda true /model/path/best_model.pth.tar model/config/path/config.json dataset/path/ output_path``` . This code parses all .wav files at the given dataset path and generates the same folder structure under the output path with the generated embedding files.
- Watch training on Tensorboard as in TTS