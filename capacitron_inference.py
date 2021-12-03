'''
This will be deleted later, only for dev, to see how to infer the capacitron model

'''

from TTS.utils.synthesizer import Synthesizer

# Sample from the prior or the posterior
prior = False

# If posterior, define the reference wav and text here
capacitron_reference_wav = None if prior else "reference_path"
capacitron_reference_text = None if prior else "transcription of the reference wav"

model = "coqui_tts-December-03-2021_12+40AM-4d4f6160"
config_path = "/home/big-boy/Models/CPR/{}/config.json".format(model)
model_path = "/home/big-boy/Models/CPR/{}/checkpoint_40000.pth.tar".format(model)

text = 'Text to synth'

out_path = "Capacitron_{}_output.wav".format('prior' if prior else 'posterior')
speakers_file_path = None
vocoder_path = None
vocoder_config_path = None
# vocoder_path = "~/Models/BlizzardVocoder/hifigan-blizzard-fine-tuning-May-31-2021_05+55PM-53c9310/checkpoint_930000.pth.tar"
# vocoder_config_path = "~/Models/BlizzardVocoder/hifigan-blizzard-fine-tuning-May-31-2021_05+55PM-53c9310/config.json"
encoder_path = None
encoder_config_path = None
speaker_idx = None
speaker_wav = None

def main():
    # load models
    synthesizer = Synthesizer(
        model_path,
        config_path,
        speakers_file_path,
        vocoder_path,
        vocoder_config_path,
        encoder_path,
        encoder_config_path,
        use_cuda=False,
    )


    # check the arguments against a multi-speaker model.
    if synthesizer.tts_speakers_file and (not speaker_idx and not speaker_wav):
        print(
            " [!] Looks like you use a multi-speaker model. Define `--speaker_idx` to "
            "select the target speaker. You can list the available speakers for this model by `--list_speaker_idxs`."
        )

    # RUN THE SYNTHESIS
    print(" > Text: {}".format(text))

    # kick it
    wav = synthesizer.tts(text, speaker_idx, speaker_wav, reference_wav=capacitron_reference_wav, reference_text=capacitron_reference_text)

    # save the results
    print(" > Saving output to {}".format(out_path))
    synthesizer.save_wav(wav, out_path)

if __name__ == "__main__":
    main()