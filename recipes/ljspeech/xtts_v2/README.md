
# Coqui AI TTS

This repository contains the code for training a text-to-speech (TTS) model using Coqui AI's TTS framework.

## Installation and Configuration

1. **Clone the repository:**

   ```bash
   git clone https://github.com/owos/coqui-ai-TTS.git
   ```

2. **Navigate to the repository root:**

   ```bash
   cd coqui-ai-TTS
   ```
3. **Create a virtual environment with python version 3.10**

   ```bash
   conda create -n xtts python==3.10
   conda activate xtts
   ```

4. **Install system dependencies and the code:**

   ```bash
   make system-deps  # Intended to be used on Ubuntu (Debian). Let us know if you have a different OS.
   make install
   ```

5. **Open the following file and redefine the specified variables:**

   File: `recipes/ljspeech/xtts_v2/train_gpt_xtts.py`

   ```python
   # Line 30
   path = 'the root path to the audio dirs on your machine'

   # Line 31
   meta_file_train = "the root path to the train CSV on your machine"

   # Line 32
   meta_file_val = "the root path to the train CSV on your machine"

   # Line 75
   SPEAKER_REFERENCE = "a list with a single path to a test audio from the afro tts data"
   ```

## Running the Code

From the repository root, run the following command:

```bash python
python3 recipes/ljspeech/xtts_v2/train_gpt_xtts.py
```

You are now ready to train your TTS model using Coqui AI's framework. Enjoy!

## Optional: Resuming from a checkpoint

File: `recipes/ljspeech/xtts_v2/train_gpt_xtts_resume.py`

Update the parameters in the file for the models


