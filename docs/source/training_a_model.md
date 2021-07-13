# Training a Model

1. Decide what model you want to use.

    Each model has a different set of pros and cons that define the run-time efficiency and the voice quality. It is up to you to decide what model servers your needs. Other than referring to the papers, one easy way is to test the 🐸TTS
    community models and see how fast and good each of the models. Or you can start a discussion on our communication channels.

2. Understand the configuration class, its fields and values of your model.

    For instance, if you want to train a `Tacotron` model then see the `TacotronConfig` class and make sure you understand it.

3. Go to the recipes and check the recipe of your target model.

    Recipes do not promise perfect models but they provide a good start point for `Nervous Beginners`. A recipe script training
    a `GlowTTS` model on `LJSpeech` dataset looks like below. Let's be creative and call this script `train_glowtts.py`.

    ```python
    # train_glowtts.py

    import os

    from TTS.tts.configs import GlowTTSConfig
    from TTS.tts.configs import BaseDatasetConfig
    from TTS.trainer import init_training, Trainer, TrainingArgs


    output_path = os.path.dirname(os.path.abspath(__file__))
    dataset_config = BaseDatasetConfig(name="ljspeech", meta_file_train="metadata.csv", path=os.path.join(output_path, "../LJSpeech-1.1/"))
    config = GlowTTSConfig(
        batch_size=32,
        eval_batch_size=16,
        num_loader_workers=4,
        num_eval_loader_workers=4,
        run_eval=True,
        test_delay_epochs=-1,
        epochs=1000,
        text_cleaner="english_cleaners",
        use_phonemes=False,
        phoneme_language="en-us",
        phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),
        print_step=25,
        print_eval=True,
        mixed_precision=False,
        output_path=output_path,
        datasets=[dataset_config]
    )
    args, config, output_path, _, c_logger, tb_logger = init_training(TrainingArgs(), config)
    trainer = Trainer(args, config, output_path, c_logger, tb_logger)
    trainer.fit()
    ```

    You need to change fields of the `BaseDatasetConfig` to match your own dataset and then update `GlowTTSConfig`
    fields as you need.

 4. Run the training.

    You need to call the python training script.

    ```bash
    $ CUDA_VISIBLE_DEVICES="0" python train_glowtts.py
    ```

    Notice that you set the GPU you want to use on your system by setting `CUDA_VISIBLE_DEVICES` environment variable.
    To see available GPUs on your system, you can use `nvidia-smi` command on the terminal.

    If you like to run a multi-gpu training

    ```bash
    $ CUDA_VISIBLE_DEVICES="0, 1, 2" python TTS/bin/distribute.py --script <path_to_your_script>/train_glowtts.py
    ```

    The example above runs a multi-gpu training using GPUs `0, 1, 2`.

    The beginning of a training run looks like below.

    ```console
    > Experiment folder: /your/output_path/-Juni-23-2021_02+52-78899209
    > Using CUDA:  True
    > Number of GPUs:  1
    > Setting up Audio Processor...
    | > sample_rate:22050
    | > resample:False
    | > num_mels:80
    | > min_level_db:-100
    | > frame_shift_ms:None
    | > frame_length_ms:None
    | > ref_level_db:20
    | > fft_size:1024
    | > power:1.5
    | > preemphasis:0.0
    | > griffin_lim_iters:60
    | > signal_norm:True
    | > symmetric_norm:True
    | > mel_fmin:0
    | > mel_fmax:None
    | > spec_gain:20.0
    | > stft_pad_mode:reflect
    | > max_norm:4.0
    | > clip_norm:True
    | > do_trim_silence:True
    | > trim_db:45
    | > do_sound_norm:False
    | > stats_path:None
    | > base:10
    | > hop_length:256
    | > win_length:1024
    | > Found 13100 files in /your/dataset/path/ljspeech/LJSpeech-1.1
    > Using model: glow_tts

    > Model has 28356129 parameters

    > EPOCH: 0/1000

    > DataLoader initialization
    | > Use phonemes: False
    | > Number of instances : 12969
    | > Max length sequence: 187
    | > Min length sequence: 5
    | > Avg length sequence: 98.3403500655409
    | > Num. instances discarded by max-min (max=500, min=3) seq limits: 0
    | > Batch group size: 0.

    > TRAINING (2021-06-23 14:52:54)

    --> STEP: 0/405 -- GLOBAL_STEP: 0
        | > loss: 2.34670
        | > log_mle: 1.61872
        | > loss_dur: 0.72798
        | > align_error: 0.52744
        | > current_lr: 2.5e-07
        | > grad_norm: 5.036039352416992
        | > step_time: 5.8815
        | > loader_time: 0.0065
    ...
    ```

5. Run the Tensorboard.

    ```bash
    $ tensorboard --logdir=<path to your training directory>
    ```

6. Check the logs and the Tensorboard and monitor the training.

    On the terminal and Tensorboard, you can monitor the losses and their changes over time. Also Tensorboard provides certain figures and sample outputs.

    Note that different models have different metrics, visuals and outputs to be displayed.

    You should also check the [FAQ page](https://github.com/coqui-ai/TTS/wiki/FAQ) for common problems and solutions
    that occur in a training.

7. Use your best model for inference.

    Use `tts` or `tts-server` commands for testing your models.

    ```bash
    $ tts --text "Text for TTS" \
          --model_path path/to/checkpoint_x.pth.tar \
          --config_path path/to/config.json \
          --out_path folder/to/save/output.wav
    ```

8. Return to the step 1 and reiterate for training a `vocoder` model.

    In the example above, we trained a `GlowTTS` model, but the same workflow applies to all the other 🐸TTS models.
