# How to get the Blizzard 2013 Dataset

The Capacitron model is a variational encoder extension of standard Tacotron based models to model prosody.

To take full advantage of the model, it is advised to train the model with a dataset that contains a significant amount of prosodic information in the utterances. A tested candidate for such applications is the blizzard2013 dataset from the Blizzard Challenge, containing many hours of high quality audio book recordings.

To get a license and download link for this dataset, you need to visit the [website](https://www.cstr.ed.ac.uk/projects/blizzard/2013/lessac_blizzard2013/license.html) of the Centre for Speech Technology Research of the University of Edinburgh.

You get access to the raw dataset in a couple of days. There are a few preprocessing steps you need to do to be able to use the high fidelity dataset.

1. Get the forced time alignments for the blizzard dataset from [here](https://github.com/mueller91/tts_alignments).
2. Segment the high fidelity audio-book files based on the instructions [here](https://github.com/Tomiinek/Blizzard2013_Segmentation).