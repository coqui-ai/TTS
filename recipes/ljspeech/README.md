# 🐸💬 TTS LJspeech Recipes

For running the recipes

1. Download the LJSpeech dataset here either manually from [its official website](https://keithito.com/LJ-Speech-Dataset/) or using ```download_ljspeech.sh```.
2. Go to your desired model folder and run the training.

    Running Python files. (Choose the desired GPU ID for your run and set ```CUDA_VISIBLE_DEVICES```)
    ```terminal
    CUDA_VISIBLE_DEVICES="0" python train_modelX.py
    ```

    Running bash scripts.
    ```terminal
    bash run.sh
    ```

💡 Note that these runs are just templates to help you start training your first model. They are not optimized for the best
result. Double-check the configurations and feel free to share your experiments to find better parameters together 💪.
