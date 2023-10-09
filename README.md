This is a fork of https://github.com/coqui-ai/TTS/ , visit the main repository for installation instructions. This repository consists of baseline models for LIMMITS 24 Speech synthesis + voice cloning challenge organised as part of ICASSP 2024. More information is available at the challenge website - https://sites.google.com/view/limmits24/home.

Pretrained models
---
YourTTS Base model with 14 speaker data from challenge dataset (1 hour from each speaker) - https://huggingface.co/SYSPIN/LIMMITS24_ML_basemodel_1hr_14speakers

Track 1 - We share the base model for track 1 (no few shot fine-tuning performed) - https://huggingface.co/SYSPIN/LIMMITS24_ML_track1
Track 2 - To be shared soon
Track3 - To be shares soon

Scripts
---
Visit LIMMITS-24-Coquiai/recipes/syspin/yourtts for training and inference scripts.
All speech data is downsampled to 16Khz to have uniformity across datasets, and begin and end silences are removed. YourTTS model is trained using coqui-ai implementation.



Regarding any queries, contact sathvikudupa66@gmail.com or challenge.syspin@iisc.ac.in
