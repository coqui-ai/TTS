This is a fork of https://github.com/coqui-ai/TTS/ , visit the main repository for installation instructions. This repository consists of baseline models for LIMMITS 24 Speech synthesis + voice cloning challenge organised as part of ICASSP 2024. More information is available at the challenge website - https://sites.google.com/view/limmits24/home.

Pretrained models
---
YourTTS Base model with 14 speaker data from challenge dataset (1 hour from each speaker) - https://huggingface.co/SYSPIN/LIMMITS24_ML_basemodel_1hr_14speakers

Track 1 - We share the base model for track 1 (no few shot fine-tuning performed, though it can be done for track1) - https://huggingface.co/SYSPIN/LIMMITS24_ML_basemodel_1hr_14speakers

Track 2 - To be shared soon

Track3 - To be shared soon

Scripts
---
Visit ```LIMMITS-24-Coquiai/recipes/syspin/yourtts``` for training and inference scripts.

Steps
1. Register for the challenge
2. Download challenge dataset - https://ee.iisc.ac.in/limmitsdataset/
3. Resample all audio to 16Khz
4. Run ```LIMMITS-24-Coquiai/recipes/syspin/yourtts/data_prep.py```
5. Provide manifest and charecter paths in ```LIMMITS-24-Coquiai/recipes/syspin/yourtts/train_yourtts.py```
6. Start training
7. Infer on target speaker with ```LIMMITS-24-Coquiai/recipes/syspin/yourtts/infer_yourtts.sh```


Regarding any queries, contact sathvikudupa66@gmail.com or challenge.syspin@iisc.ac.in
