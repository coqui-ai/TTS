## Utilities to Convert Models to Tensorflow2
Here there are experimental utilities to convert trained Torch models to Tensorflow (2.2>=). 

Converting Torch models to TF enables all the TF toolkit to be used for better deployment and device specific optimizations.

Note that we do not plan to share training scripts for Tensorflow in near future. But any contribution in that direction would be more than welcome.

To see how you can use TF model at inference, check the notebook.

This is an experimental release. If you encounter an error, please put an issue or in the best send a PR but you are mostly on your own.


### Converting a Model
- Run ```convert_tacotron2_torch_to_tf.py --torch_model_path /path/to/torch/model.pth.tar --config_path /path/to/model/config.json --output_path /path/to/output/tf/model``` with the right arguments.

### Known issues ans limitations
- We use a custom model load/save mechanism which enables us to store model related information with models weights. (Similar to Torch). However, it is prone to random errors.
- Current TF model implementation is slightly slower than Torch model. Hopefully, it'll get better with improving TF support for eager mode and ```tf.function```.
- TF implementation of Tacotron2 only supports regular Tacotron2 as in the paper.
- You can only convert models trained after TF model implementation since model layers has been updated in Torch model.
