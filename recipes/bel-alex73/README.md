This description was created based on [jhlfrfufyfn/ml-bel-tts](https://github.com/jhlfrfufyfn/ml-bel-tts). Thanks a lot to jhlfrfufyfn for advices, configuration, code and ideas.

# Training

This recipe uses [CommonVoice](https://commonvoice.mozilla.org/en/datasets) dataset. It has format mp3/32kHz/48kbps format and contains multiple speakers because it was created for voice recognition. Looks like it's the best voice corpus of Belarussian language for today. But for creating better voice synthesis it will require to record some specific corpus with good pronunciation and good record quality.

Looks like for Belarusian Common Voice corpus there is no sense to train full big dataset (90 hours). It's enough 30 hours dataset, that makes very good progress for 350 epochs(24000 steps on 24GiB GPU). The quality of dataset is more important that size.

To train a model, you need to:
- download code and data
- prepare training data and generate scale_stats file
- change configuration settings
- train TTS model (GlowTTS in this example)
- train Vocoder model (HiFiGAN in this example)

We recommend to prepare all things locally, then train models on the external computer with fast GPU. Text below describes all these steps.

## Download code and data

It would be good to place all things into local folder like /mycomputer/. You need files:

- Coqui-TTS - code from this git. For example, to /mycomputer/TTS/. *Expected result: you have /mycomputer/TTS/setup.py and other files from git.*
- [Common voice dataset](https://commonvoice.mozilla.org/en/datasets) into cv-corpus/ directory near Coqui-TTS. *Expected result: you have /mycomputer/cv-corpus/be/validated.tsv and more than 1 mln .mp3 files in the /mycomputer/cv-corpus/be/clips/.*
- Belarusian text to phonemes converter - fanetyka.jar from the [https://github.com/alex73/Software-Korpus/releases](https://github.com/alex73/Software-Korpus/releases), then place it to fanetyka/ near Coqui-TTS. *Expected result: you have file /mycomputer/fanetyka/fanetyka.jar*

Prepared data will be stored into storage/ directory near Coqui-TTS, like /mycomputer/storage/.

## Prepare to training - locally

Docker container was created for simplify local running. You can run `docker-prepare-start.sh` to start environment. All commands below should be started in docker console.

* Start jupyter by the command `jupyter notebook --no-browser --allow-root --port=2525 --ip=0.0.0.0`. It will display link to http. You need to open this link, then choose `recipes/bel-alex73/choose_speaker.ipynb` notebook. You should run cells one-by-one, listen different speakers and select speaker that you want to use. After all commands in notebook, you can press Ctrl+C in docker console to stop jupyter. *Expected result: directory /mycomputer/storage/filtered_dataset/ with df_speaker.csv file and many *.wav files.*

* Convert text to phonemes: `java -cp /a/fanetyka/fanetyka.jar org.alex73.fanetyka.impl.FanetykaTTSPrepare /storage/filtered_dataset/df_speaker.csv /storage/filtered_dataset/ipa_final_dataset.csv`. It will display all used characters at the end. You can use these characters to modify config in train_glowtts.py. *Expected result: file /mycomputer/storage/filtered_dataset/ipa_final_dataset.csv*

* Modify configs(if you need) in the train_glowtts.py and train_hifigan.py. Then export config to old json format to create scale_stats.npy by the command `python3 recipes/bel-alex73/dump_config.py > recipes/bel-alex73/config.json`. *Expected result: file /mycomputer/TTS/recipes/bel-alex73/config.json exists.*

* Start scale_stats.npy, that will the model to learn better: `mkdir -p /storage/TTS/; python3 TTS/bin/compute_statistics.py --config_path recipes/bel-alex73/config.json --out_path /storage/TTS/scale_stats.npy`. *Expected result: file /mycomputer/storage/TTS/scale_stats.npy exists.*

## Training - with GPU

You need to upload Coqui-TTS(/mycomputer/TTS/) and storage/ directory(/mycomputer/storage/) to some computer with GPU. We don't need cv-corpus/ and fanetyka/ directories for training. Install gcc, then run `pip install -e .[all,dev,notebooks]` to prepare modules. GlowTTS and HifiGan models should be learned separately based on /storage/filtered_dataset only, i.e. they are not dependent from each other. <devices> below means list of GPU ids from zero("0,1,2,3" for systems with 4 GPU). See details on the https://tts.readthedocs.io/en/latest/tutorial_for_nervous_beginners.html(multi-gpu training).

Current setup created for 24GiB GPU. You need to change batch_size if you have more or less GPU memory. Also, you can try to set lr(learning rate) to lower value in the end of training GlowTTS.

* Start GlowTTS model training by the command `OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=<devices> python3 -m trainer.distribute --script recipes/bel-alex73/train_glowtts.py`. It will produce training data into storage/output/ directory. Usually 100.000 global steps required. *Expected behavior: You will see /storage/output-glowtts/<start_date>/best_model_<step>.pth files.*

* Start HiFiGAN model training by the command `OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=<devices> python3 -m trainer.distribute --script recipes/bel-alex73/train_hifigan.py`. *Expected behavior: You will see /storage/output-hifigan/<start_date>/best_model_<step>.pth files.*

## How to monitor training

* Run `nvidia-smi` to be sure that training uses all GPUs and to be sure that you are using more than 90% GPU memory and utilization.

* Run `tensorboard --logdir=/storage/output-<model>/` to see alignment, avg_loss metrics and check audio evaluation. You need only events.out.tfevents.\* files for that.

## Synthesizing speech

	tts --text "<phonemes>" --out_path output.wav \
		--config_path /storage/output-glowtts/run/config.json \
		--model_path /storage/output-glowtts/run/best_model.pth \
		--vocoder_config_path /storage/output-hifigan/run/config.json \
		--vocoder_path /storage/output-hifigan/run/best_model.pth
