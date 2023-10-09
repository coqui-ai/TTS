#!/bin/bash

folder="<checkpoint-folder>"
chk=$folder/"<checkpoint-name>"

text="केंद्र और राज्यों द्वारा समान कर छूट सूची में ब्रेड, अंडे, दूध, सब्जियां, अनाज, किताबें और नमक शामिल हैं।" #hindi

reference_wav="<path to reference wav file of target speaker>"

config=$folder/"config.json"

savepath="<save path for generated audio>"

tts --text "$text" --speaker_wav "$ref_wav" --model_path $chk --config_path $config --out_path $savepath --language_idx "Hindi"
