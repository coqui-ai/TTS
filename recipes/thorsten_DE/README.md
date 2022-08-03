# 🐸💬 TTS Thorsten Recipes

For running the recipes you need the [Thorsten-Voice](https://github.com/thorstenMueller/Thorsten-Voice) dataset.

You can download it manually from [the official website](https://www.thorsten-voice.de/) or use ```download_thorsten_de.sh``` alternatively running any of the **train_modelX.py**scripts will download the dataset if not already present.

Then, go to your desired model folder and run the training.

    Running Python files. (Choose the desired GPU ID for your run and set ```CUDA_VISIBLE_DEVICES```)
    ```terminal
    CUDA_VISIBLE_DEVICES="0" python train_modelX.py
    ```

💡 Note that these runs are just templates to help you start training your first model. They are not optimized for the best
result. Double-check the configurations and feel free to share your experiments to find better parameters together 💪.
