## Utilities to Convert Models to Tensorflow2
Here there are utilities to convert trained Torch models to Tensorflow (2.2>=). 

We currently support Tacotron2 with Location Sensitive Attention.

Be aware that our old Torch models may not work with this module due to additional changes in layer naming convention. Therefore, you need to train new models or handle these changes.

We do not plan to share training scripts for Tensorflow in near future. But any contribution in that direction would be more than welcome.

To see how you can use TF model at inference, check the notebook.

This is an experimental release. If you encounter an error, please put an issue or in the best send a PR but you are mostly on your own.
